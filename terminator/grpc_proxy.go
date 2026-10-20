// Package main — gRPC L7 代理。
//
// 不使用 L4 透传：L4 会让 Python 侧看不到客户端证书（mTLS 已在终结器结束），
// 也没有地方注入身份 metadata。这里用 UnknownServiceHandler 逐流转发原始
// protobuf 消息，保留 unary / 客户端流 / 服务端流 / 双向流、header、trailer
// 与 metadata，并在转发前剥离客户端伪造的身份 metadata。
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"time"

	"github.com/fengzhizi319/PrivShield-go/pkg/tlsutil"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"
)

// rawMessage 承载未经解析的 protobuf 消息体。
type rawMessage struct {
	data []byte
}

// rawCodec 让终结器完全不依赖 proto 定义：消息按原始字节收发。
// Name 保持 "proto"，避免改变 content-subtype 与下游兼容性。
type rawCodec struct{}

func (rawCodec) Marshal(value any) ([]byte, error) {
	switch msg := value.(type) {
	case *rawMessage:
		return msg.data, nil
	case rawMessage:
		return msg.data, nil
	default:
		return nil, fmt.Errorf("rawCodec: unsupported marshal type %T", value)
	}
}

func (rawCodec) Unmarshal(data []byte, value any) error {
	msg, ok := value.(*rawMessage)
	if !ok {
		return fmt.Errorf("rawCodec: unsupported unmarshal type %T", value)
	}
	msg.data = append(msg.data[:0], data...)
	return nil
}

func (rawCodec) Name() string { return "proto" }

type grpcProxy struct {
	conn   *grpc.ClientConn
	signer *identitySigner
}

func newGRPCProxy(upstream string, signer *identitySigner) (*grpcProxy, error) {
	conn, err := grpc.NewClient(upstream,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.ForceCodec(rawCodec{}),
			grpc.MaxCallRecvMsgSize(maxGRPCMessageSize),
			grpc.MaxCallSendMsgSize(maxGRPCMessageSize),
		),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                2 * time.Minute,
			Timeout:             20 * time.Second,
			PermitWithoutStream: true,
		}),
	)
	if err != nil {
		return nil, fmt.Errorf("dial gRPC upstream %s: %w", upstream, err)
	}
	return &grpcProxy{conn: conn, signer: signer}, nil
}

func (p *grpcProxy) Close() error {
	if p == nil || p.conn == nil {
		return nil
	}
	return p.conn.Close()
}

// handler 是 UnknownServiceHandler：所有 RPC（含 health / reflection）逐流转发。
func (p *grpcProxy) handler(_ any, stream grpc.ServerStream) error {
	ctx := stream.Context()
	fullMethod, ok := grpc.Method(ctx)
	if !ok || fullMethod == "" {
		return status.Error(codes.Internal, "terminator: missing gRPC method")
	}

	var identity *tlsutil.PeerIdentity
	if grpcPeer, ok := peer.FromContext(ctx); ok && grpcPeer != nil && grpcPeer.AuthInfo != nil {
		identity = tlsutil.PeerFromGRPCAuthInfo(grpcPeer.AuthInfo)
	}

	outMD := metadata.MD{}
	if incoming, ok := metadata.FromIncomingContext(ctx); ok {
		outMD = incoming.Copy()
	}
	stripGRPCIdentityMetadata(outMD)
	if p.signer != nil {
		for key, values := range p.signer.grpcMetadata(fullMethod, identity) {
			outMD.Set(key, values...)
		}
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	outCtx := metadata.NewOutgoingContext(ctx, outMD)

	clientStream, err := p.conn.NewStream(outCtx, &grpc.StreamDesc{
		// 以 bidi 描述创建流即可覆盖全部四种 RPC 形态：流方向由实际
		// SendMsg/RecvMsg 决定，终结器不需要知道方法签名。
		ServerStreams: true,
		ClientStreams: true,
	}, fullMethod)
	if err != nil {
		return grpcStatusError(err)
	}

	sendErr := make(chan error, 1)
	go func() {
		for {
			var msg rawMessage
			if err := stream.RecvMsg(&msg); err != nil {
				if errors.Is(err, io.EOF) {
					sendErr <- clientStream.CloseSend()
					return
				}
				sendErr <- err
				cancel()
				return
			}
			if err := clientStream.SendMsg(&msg); err != nil {
				sendErr <- err
				cancel()
				return
			}
		}
	}()

	header, err := clientStream.Header()
	if err != nil {
		select {
		case upstreamErr := <-sendErr:
			if upstreamErr != nil {
				return grpcStatusError(upstreamErr)
			}
		default:
		}
		return grpcStatusError(err)
	}
	if len(header) > 0 {
		if err := stream.SendHeader(header); err != nil {
			return grpcStatusError(err)
		}
	}

	for {
		var msg rawMessage
		if err := clientStream.RecvMsg(&msg); err != nil {
			if errors.Is(err, io.EOF) {
				stream.SetTrailer(clientStream.Trailer())
				return nil
			}
			slog.Debug("upstream gRPC stream terminated", "method", fullMethod, "err", err)
			return grpcStatusError(err)
		}
		if err := stream.SendMsg(&msg); err != nil {
			return grpcStatusError(err)
		}
	}
}

// grpcStatusError 保留上游 status code（尤其是 Unavailable），只把非 status
// 错误折叠为 Unavailable —— 绝不能把"后端不可达"降级成 Internal。
func grpcStatusError(err error) error {
	if err == nil {
		return nil
	}
	if _, ok := status.FromError(err); ok {
		return err
	}
	switch {
	case errors.Is(err, context.Canceled):
		return status.Error(codes.Canceled, "request canceled")
	case errors.Is(err, context.DeadlineExceeded):
		return status.Error(codes.DeadlineExceeded, "deadline exceeded")
	default:
		return status.Error(codes.Unavailable, err.Error())
	}
}

// newGRPCServer 装配 mTLS gRPC 服务：业务口协议栈 + CN 白名单拦截器 + 原始
// 消息代理。白名单为空时不挂拦截器，身份裁决回落到 Python 引擎。
func newGRPCServer(
	built *tlsutil.BuiltStack,
	cfg *config,
	signer *identitySigner,
) (*grpc.Server, *grpcProxy, *tlsutil.DynamicWhitelist, error) {
	creds, err := built.GRPCServerCredentials()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("build gRPC server credentials: %w", err)
	}
	proxy, err := newGRPCProxy(cfg.grpcUpstream, signer)
	if err != nil {
		return nil, nil, nil, err
	}

	opts := []grpc.ServerOption{
		grpc.Creds(creds),
		grpc.ForceServerCodec(rawCodec{}),
		grpc.UnknownServiceHandler(proxy.handler),
		grpc.MaxRecvMsgSize(maxGRPCMessageSize),
		grpc.MaxSendMsgSize(maxGRPCMessageSize),
		grpc.KeepaliveParams(keepalive.ServerParameters{
			MaxConnectionIdle: 5 * time.Minute,
			MaxConnectionAge:  2 * time.Hour,
			Time:              2 * time.Minute,
			Timeout:           20 * time.Second,
		}),
		grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{
			MinTime:             5 * time.Second,
			PermitWithoutStream: true,
		}),
	}

	var whitelist *tlsutil.DynamicWhitelist
	if cfg.whitelistFile != "" {
		unary, stream, dw, err := tlsutil.NewWhitelistInterceptor(cfg.whitelistFile)
		if err != nil {
			_ = proxy.Close()
			return nil, nil, nil, fmt.Errorf("init mTLS whitelist interceptor: %w", err)
		}
		if unary == nil || stream == nil {
			_ = proxy.Close()
			return nil, nil, nil, fmt.Errorf("mTLS whitelist interceptor was not registered (path=%s)", cfg.whitelistFile)
		}
		opts = append(opts,
			grpc.ChainUnaryInterceptor(unary),
			grpc.ChainStreamInterceptor(stream),
		)
		whitelist = dw
	}
	return grpc.NewServer(opts...), proxy, whitelist, nil
}

#!/usr/bin/env python3
"""医疗数据生成脚本 / Medical Data Generator.

生成 20 条逼真的模拟病人医疗记录 CSV（kangyang.csv，康养数据），
用于分类分级与脱敏流水线测试。

Usage:
    python scripts/data/generate_medical_data.py
    python scripts/data/generate_medical_data.py --output data/kangyang.csv --count 20
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from datetime import date, datetime, timedelta
from pathlib import Path

# === 数据池 ===

SURNAMES = list("李王张刘陈杨赵黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕")
GIVEN_NAMES = [
    "维","芳","娜","秀英","敏","静","丽","强","磊","军","洋","勇","艳","杰","娟",
    "涛","明","超","秀兰","霞","平","刚","华","梅","鹏","飞","鑫","波","斌",
    "宇","浩","凯","思远","志强","建华","建国","海燕","晓明","小红","文静","丽华",
    "秀珍","建军","志明","雅琴","美玲","子涵","梓轩","浩然","欣怡","诗涵","雨泽",
]

AREA_CODES = [
    "110101","110102","110105","310101","310104","310105",
    "440103","440104","440106","510102","510104","510105",
    "330102","330103","330105","320102","320104","320106",
    "420102","420103","420106","610102","610103","610104",
    "500101","500102","500103","430102","430103","430105",
]

ADDRESSES = [
    ("北京市","东城区","东华门街道","朝阳门内大街",23),
    ("北京市","西城区","金融街街道","阜成门内大街",15),
    ("北京市","朝阳区","建外街道","建国门外大街",88),
    ("北京市","海淀区","中关村街道","中关村南大街",5),
    ("上海市","黄浦区","南京东路街道","南京东路",100),
    ("上海市","徐汇区","湖南路街道","淮海中路",1688),
    ("上海市","浦东新区","陆家嘴街道","浦东南路",500),
    ("广东省广州市","天河区","天河南街道","天河路",228),
    ("广东省广州市","越秀区","北京街道","中山五路",68),
    ("四川省成都市","锦江区","盐市口街道","东大街",36),
    ("四川省成都市","武侯区","浆洗街街道","武侯祠大街",18),
    ("浙江省杭州市","上城区","清波街道","南山路",200),
    ("浙江省杭州市","西湖区","北山街道","北山路",27),
    ("江苏省南京市","玄武区","梅园新村街道","长江路",292),
    ("湖北省武汉市","武昌区","水果湖街道","东湖路",168),
    ("陕西省西安市","碑林区","南院门街道","南大街",68),
    ("重庆市","渝中区","解放碑街道","民生路",232),
    ("湖南省长沙市","芙蓉区","定王台街道","芙蓉中路",456),
    ("山东省济南市","历下区","泉城路街道","泉城路",180),
    ("辽宁省沈阳市","沈河区","大南街道","大南街",218),
]

DIAGNOSES = [
    ("2型糖尿病", "内分泌科", "多饮多尿多食伴体重下降3月"),
    ("原发性高血压(3级 极高危)", "心内科", "反复头晕伴血压升高2年"),
    ("冠心病 不稳定型心绞痛", "心内科", "反复胸闷胸痛半年，加重1周"),
    ("社区获得性肺炎(重症)", "呼吸内科", "发热咳嗽咳痰5天"),
    ("慢性阻塞性肺疾病急性加重", "呼吸内科", "反复咳嗽咳痰10年，气促3年，加重2天"),
    ("腰椎间盘突出症(L4/5)", "骨科", "腰痛伴右下肢放射痛2月"),
    ("左股骨颈头下型骨折", "骨科", "摔伤致左髋部疼痛、活动受限3小时"),
    ("胃窦低分化腺癌", "肿瘤科", "上腹部隐痛伴消瘦3月"),
    ("甲状腺乳头状癌", "肿瘤科", "发现颈部肿物1月"),
    ("急性脑梗死(左侧基底节区)", "神经内科", "突发右侧肢体无力4小时"),
    ("帕金森病(Hoehn-Yahr 3级)", "神经内科", "双手震颤伴行动迟缓2年"),
    ("重度抑郁发作", "精神科", "情绪低落、失眠、兴趣减退半年，自杀意念2周"),
    ("精神分裂症(偏执型)", "精神科", "幻听、被害妄想反复发作3年，再发1月"),
    ("一期梅毒(硬下疳)", "皮肤性病科", "外阴出现无痛性溃疡1周"),
    ("淋菌性尿道炎", "皮肤性病科", "尿频尿急尿痛伴尿道口溢脓3天"),
    ("尖锐湿疣", "皮肤性病科", "发现外阴菜花状赘生物2月"),
    ("获得性免疫缺陷综合征(HIV/AIDS)", "感染性疾病科", "反复发热盗汗2月伴体重消瘦10kg"),
    ("慢性乙型病毒性肝炎(肝硬化期)", "肝病科", "反复乏力纳差5年，加重伴腹水1周"),
    ("遗传性亨廷顿舞蹈病", "神经遗传科", "不自主舞蹈样肢体抽搐伴认知下降1年"),
    ("BRCA1突变相关遗传性乳腺癌", "乳腺外科", "发现左乳无痛性肿块2月"),
    ("类风湿关节炎(活动期)", "风湿免疫科", "双手指间关节肿痛晨僵1年"),
    ("慢性肾脏病3期", "肾内科", "发现血肌酐升高1年，伴双下肢水肿"),
    ("缺铁性贫血(重度)", "血液科", "乏力、面色苍白3月，活动后心悸气短"),
    ("急性化脓性阑尾炎", "普外科", "转移性右下腹痛12小时"),
    ("双眼老年性白内障", "眼科", "双眼视力渐进性下降1年"),
    ("支气管哮喘(重度持续)", "呼吸内科", "反复发作性喘息5年，再发伴呼吸困难2天"),
    ("重度骨质疏松症(T值-3.8)", "骨科", "腰背痛伴身高缩短2年，跌倒后髋部骨折"),
    ("肌萎缩侧索硬化症(ALS渐冻症)", "神经内科", "双下肢无力肌跳1年，伴咽下困难"),
]

# L4 级现病史（详细性病/肿瘤/传染病/手术记录）
PRESENT_ILLNESS_L4 = [
    "患者自述1月前出现外阴溃疡伴无痛性硬下疳。实验室检查示苍白密螺旋体特异性抗体(TPPA)阳性，快速血浆反应素试验(RPR) 1:16阳性。诊断为一期梅毒，给予苄星青霉素 240万U im qw×3周抗梅毒治疗。",
    "患者3天前出现尿频、尿急、尿痛及尿道口溢脓。局部分泌物涂片发现革兰氏阴性双球菌，淋球菌核酸检测(PCR)阳性。诊断为淋菌性尿道炎，予大观霉素 2g im 单次给药。",
    "患者2月前发现外阴及肛周多发菜花状赘生物，伴异物感与接触性出血。病理活检提示尖锐湿疣，HPV 16/18高危基因型阳性。行CO2激光灼除治疗+咪喹莫特乳膏外用。",
    "患者3月前出现上腹部持续性隐痛，进食后加重。胃镜示：胃窦部3.0cm×2.5cm溃疡性肿物，病理示低分化腺癌。腹部CT示胃窦壁增厚，周围多发肿大淋巴结。于2024年3月15日在全麻下行腹腔镜辅助远端胃大部切除术(Billroth-II式)，术中见肿瘤约3cm，未侵及浆膜。术后行FOLFOX方案辅助化疗6周期。",
    "患者3月来反复咳嗽、痰中带血。胸部CT示右肺上叶3.8cm占位伴纵隔淋巴结转移。骨扫描示多发骨转移。穿刺活检示肺浸润性腺癌，EGFR L858R突变，口服阿美替尼110mg qd靶向治疗。",
    "慢性乙型病毒性肝炎病史12年，HBV-DNA 4.8×10⁶ IU/mL。腹部超声与CT提示肝硬化代偿期、门静脉高压、脾大。口服恩替卡韦0.5mg qd抗病毒，肝功能ALT 88U/L，AST 95U/L。",
    "患者4小时前摔倒，左髋部着地，X线示左股骨颈头下型骨折Garden III型。于2024年5月20日在腰硬联合麻醉下行左侧人工股骨头置换术。术中取左髋外侧切口12cm，见股骨颈骨折断端，取出股骨头直径46mm，安装假体柄于股骨髓腔，复位满意。",
    "患者4小时前突发右侧肢体无力伴言语含糊。头颅CT排除脑出血，NIHSS评分12分。于发病3.5h内予rt-PA 0.9mg/kg静脉溶栓，24h后复查头颅MRI示左侧基底节区急性脑梗死。DSA示左侧大脑中动脉狭窄，予双抗+调脂治疗。",
    "患者半年前发现右颈部一约2cm肿物，FNAC示甲状腺乳头状癌。于2024年4月10日在全麻下行右侧甲状腺腺叶+峡部切除+中央区淋巴结清扫术。术后予左甲状腺素钠片100μg qd替代+TSH抑制治疗。",
    "患者发现宫颈高危型HPV 16阳性1年，阴道镜下活检提示宫颈鳞状细胞癌(IB1期)。行腹腔镜下全子宫切除+双侧附件切除+盆腔淋巴结清扫术，术后顺铂周方案同步放化疗。",
]

# L5 级现病史（HIV/重度精神障碍/基因缺陷等极高敏感信息）
PRESENT_ILLNESS_L5 = [
    "患者因反复发热2月、盗汗、体重骤降10kg就诊。血清HIV-1/2抗体初筛及确证试验均呈阳性，外周血CD4+ T细胞计数 115/μL。确诊为获得性免疫缺陷综合征(HIV/AIDS)，启动三联抗逆转录病毒治疗(ART: 替诺福韦+拉米夫定+多替拉韦)。",
    "精神分裂症病史10年。近2周病情复发，出现严重命令性幻听与被害妄想（怀疑邻居通过无线电对其进行精神控制），伴冲动砸物与自伤倾向。紧急收入精神卫生中心，给予保护性约束及奥氮平 20mg/d 联合氨磺必利治疗。",
    "携带遗传性亨廷顿舞蹈病致病基因(HTT基因CAG重复序列46次)，家族中父亲与长兄均确诊并已故。患者呈现典型不自主舞蹈样躯体动作与重度认知功能障碍(MoCA 11/30)，予四苯嗪及心理支持。",
    "患者基因检测示BRCA1基因185delAG突变阳性，一级亲属中母亲患卵巢癌、姨母患乳腺癌。全外显子组测序(WES)发现 additional pathogenic variant in TP53 (R213*)。经遗传咨询后患者知情同意行预防性双侧乳腺+卵巢输卵管切除术。",
    "患者精神分裂症病史15年，曾因多次冲动行为被强制住院治疗(共计7次)。本次发作期间出现命令性幻听，伴严重被害妄想。既往有2次自杀未遂史，3次伤人史。目前使用氯氮平600mg/d+利培酮4mg/d联合治疗，HIV抗体筛查阳性。",
    "患者基因检测提示渐冻症致病突变(SOD1基因c.272A>C p.D91A)，出现双下肢肌无力伴肌肉跳动1.5年，近期扩展至双上肢与吞咽困难。给予利鲁唑 50mg bid 延缓病程发展。",
]

# L3 级现病史（普通病史）
PRESENT_ILLNESS_L3 = [
    "患者3月来无明显诱因出现口渴、多饮，每日饮水量约3000mL，伴多尿、多食，体重下降约5kg。自测空腹血糖12.3mmol/L。门诊查HbA1c 9.8%，空腹C肽0.8ng/mL。诊断为2型糖尿病，予二甲双胍500mg bid+格列美脲2mg qd治疗，血糖控制欠佳。",
    "患者2年来反复出现头晕，测血压最高180/110mmHg，平素口服氨氯地平5mg qd，血压波动在150-160/90-100mmHg。1周来头晕加重，伴视物模糊。门诊查血压172/105mmHg，心电图示左室高电压，心脏超声示室间隔增厚(12mm)，左房增大(42mm)。",
    "患者半年来反复出现胸骨后压榨性疼痛，多于快走或上楼时发作，休息3-5分钟可缓解。1周来发作频率增加，每日2-3次，持续时间延长至10-15分钟。门诊心电图示V1-V4 ST段压低0.1-0.2mV。冠脉CTA示前降支近段狭窄约75%。",
    "患者5天前受凉后出现发热，体温最高38.9°C，伴咳嗽、咳黄色粘痰。血常规示WBC 13.2×10⁹/L，N% 82%。CRP 56mg/L。胸部CT示右下肺大片实变影，内见支气管充气征。PCT 2.8ng/mL。",
    "患者10年来反复咳嗽、咳白色粘痰，每年持续3月以上。3年来出现活动后气促，爬2层楼即感呼吸困难。近2天受凉后上述症状加重。肺功能示FEV1/FVC 52%，FEV1占预计值45%(GOLD 3级)。",
]

PAST_HISTORIES = [
    "既往体健，否认高血压、糖尿病、冠心病等慢性病史。否认肝炎、结核等传染病史。10年前因'急性阑尾炎'行阑尾切除术，术后恢复良好。",
    "高血压病史8年，最高180/110mmHg，口服缬沙坦80mg qd，血压控制可。2型糖尿病史5年，口服二甲双胍500mg bid，空腹血糖波动在7-9mmol/L。",
    "冠心病史3年，2年前因'急性心肌梗死'行PCI术，植入药物洗脱支架2枚。术后规律服用阿司匹林+氯吡格雷+阿托伐他汀。",
    "慢性乙肝病史15年，口服恩替卡韦0.5mg qd抗病毒治疗，HBV-DNA低于检测下限。否认肝硬化病史。",
    "既往体健。否认手术外伤史。否认输血史。预防接种史随当地计划。",
    "哮喘病史10年，间断使用沙丁胺醇气雾剂。过敏性鼻炎史5年。否认其他慢性病史。",
    "肺结核病史5年，经规范抗结核治疗9个月后治愈。否认其他传染病史。",
    "高脂血症病史3年，口服瑞舒伐他汀10mg qn。高尿酸血症病史2年，未规律用药。",
]

PERSONAL_HISTORY_SMOKERS = [
    "吸烟30年，每日20-30支(约30包年)，未戒烟。饮酒20年，每日白酒约100mL。",
    "吸烟20年，每日10-15支(约15包年)，已戒烟2年。偶尔饮酒。",
    "吸烟15年，每日1包，未戒烟。否认饮酒史。",
    "吸烟40年，每日30支(约60包年)，未戒烟。饮酒40年，每日白酒约200mL。",
]

PERSONAL_HISTORY_NON_SMOKERS = [
    "否认吸烟史。偶尔饮酒，已戒酒5年。否认疫区接触史。",
    "否认吸烟饮酒史。规律运动，每周3次，每次30分钟。",
    "否认吸烟史。少量饮酒。否认毒物及放射线接触史。",
    "否认吸烟饮酒史。否认特殊药物过敏史(另见过敏史栏)。职业为教师，否认粉尘及化学毒物接触。",
]

FAMILY_HISTORIES = [
    "父亲患高血压，母亲患2型糖尿病，均健在。否认家族性遗传病史。",
    "父亲因'胃癌'去世(65岁)，母亲健在。一弟患'2型糖尿病'。否认其他家族遗传病史。",
    "父母均健在，体健。否认家族肿瘤病史。否认近亲婚配史。",
    "母亲患'乳腺癌'(58岁确诊)，外婆患'卵巢癌'。家族中多人携带BRCA基因突变。",
    "父亲50岁时因'脑卒中'去世，母亲患'阿尔茨海默病'。一兄患'帕金森病'。",
    "否认家族遗传病史。否认家族肿瘤病史。",
]

ALLERGIC_HISTORIES = [
    "否认食物及药物过敏史。",
    "青霉素过敏(皮疹)，磺胺类药物过敏(荨麻疹)。",
    "海鲜过敏(荨麻疹)，花粉过敏(过敏性鼻炎)。",
    "头孢类抗生素过敏(过敏性休克史1次)，磺胺类过敏。",
    "磺胺类药物过敏。否认食物过敏。",
    "芒果过敏(口唇肿胀)。否认药物过敏。",
]

DISABILITY_CATEGORIES = [
    ("肢体残疾", "一级", "肢体功能综合评估", "完全不能独立生活，需完全护理", "35"),
    ("肢体残疾", "二级", "肢体功能综合评估", "大部分不能独立生活，需大部分护理", "50"),
    ("肢体残疾", "三级", "肢体功能综合评估", "部分不能独立生活，需部分护理", "65"),
    ("视力残疾", "一级", "视力功能评估", "无光感或光感", "20"),
    ("听力残疾", "二级", "听力功能评估", "重度听力损失", "40"),
    ("智力残疾", "二级", "智力功能综合评估", "重度智力低下，生活大部分不能自理", "45"),
    ("精神残疾", "一级", "精神功能综合评估", "适应行为严重障碍，生活完全不能自理", "30"),
    ("言语残疾", "二级", "言语功能评估", "完全不能进行言语交流", "55"),
]

PROGRESS_NOTE_TEMPLATES = [
    "今日查房：患者神志清楚，精神可。诉{complaint}较前好转。查体：T {temp}°C，P {pulse}次/分，R {resp}次/分，BP {bp}mmHg。{exam_findings}。目前治疗方案继续执行，注意观察病情变化。",
    "今日查房：患者诉{complaint}。查体所见同前。辅助检查回报：{lab_results}。经科室讨论，调整治疗方案如下：{plan}。已向患者及家属交代病情。",
    "术后第{day}天查房：患者生命体征平稳，诉切口轻度疼痛(NRS {nrs}分)。查体：切口敷料干燥，无渗出，周围无红肿。引流管通畅，引流量约{drain}mL，色暗红。继续予抗感染、镇痛、营养支持治疗。鼓励患者床上活动。",
]


# === 工具函数 ===

def _id_card_checksum(id17: str) -> str:
    """计算身份证第18位校验码 (MOD 11-2)。"""
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    check_map = "10X98765432"
    total = sum(int(id17[i]) * weights[i] for i in range(17))
    return check_map[total % 11]


def gen_id_card() -> str:
    """生成符合 GB 11643 的18位身份证号。"""
    area = random.choice(AREA_CODES)
    year = random.randint(1945, 2005)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    seq = random.randint(1, 999)
    id17 = f"{area}{year:04d}{month:02d}{day:02d}{seq:03d}"
    return id17 + _id_card_checksum(id17)


def gen_name() -> str:
    return random.choice(SURNAMES) + random.choice(GIVEN_NAMES)


def gen_address() -> str:
    prov, city, street, road, num = random.choice(ADDRESSES)
    building = random.randint(1, 30)
    unit = random.randint(1, 6)
    room = random.randint(101, 2501)
    return f"{prov}{city}{street}{road}{num}号{building}栋{unit}单元{room}室"


def gen_disability_cert() -> str:
    """生成残疾证号：地区码(6) + 类别(2) + 等级(1) + 序号(4) + 校验(1)。"""
    area = random.choice(AREA_CODES)
    cat_code = f"{random.randint(1,7):02d}"
    level = str(random.randint(1, 4))
    seq = f"{random.randint(1, 9999):04d}"
    body = area + cat_code + level + seq
    check = str(sum(int(c) for c in body) % 10)
    return body + check


def gen_medical_insurance() -> str:
    """生成医保证号：地区码(6) + '01' + 序号(8)。"""
    area = random.choice(AREA_CODES)
    seq = f"{random.randint(1, 99999999):08d}"
    return area + "01" + seq


def gen_date_recent(years: int = 2) -> str:
    """生成近N年内随机日期，格式 YYYY-MM-DD。"""
    end = date(2025, 6, 1)
    start = end - timedelta(days=365 * years)
    delta = (end - start).days
    d = start + timedelta(days=random.randint(0, delta))
    return d.strftime("%Y-%m-%d")


def gen_datetime_recent(years: int = 2) -> str:
    """生成近N年内随机日期时间，格式 YYYY-MM-DD HH:MM:SS。"""
    end = datetime(2025, 6, 1, 17, 0)
    start = end - timedelta(days=365 * years)
    delta_seconds = int((end - start).total_seconds())
    dt = start + timedelta(seconds=random.randint(0, delta_seconds))
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _fill_progress_note(template: str, chief: str) -> str:
    """填充病程记录模板。"""
    replacements = {
        "{complaint}": random.choice(["胸闷", "腹痛", "头痛", "咳嗽", "乏力", "伤口疼痛"]),
        "{temp}": f"{random.uniform(36.2, 38.5):.1f}",
        "{pulse}": str(random.randint(65, 110)),
        "{resp}": str(random.randint(16, 28)),
        "{bp}": f"{random.randint(110,180)}/{random.randint(60,100)}",
        "{exam_findings}": random.choice([
            "双肺呼吸音清，未闻及干湿啰音",
            "腹软，无压痛及反跳痛",
            "心率齐，各瓣膜听诊区未闻及杂音",
            "双下肢轻度凹陷性水肿",
        ]),
        "{lab_results}": random.choice([
            "血常规：WBC 8.5×10⁹/L，Hb 125g/L；肝肾功能正常",
            "血糖：空腹7.2mmol/L，餐后2h 11.3mmol/L；HbA1c 7.8%",
            "CRP 12mg/L，PCT 0.3ng/mL，较前好转",
            "甲功：TSH 3.2mIU/L，FT4 15.8pmol/L",
        ]),
        "{plan}": random.choice([
            "加用胰岛素皮下注射，监测血糖谱",
            "调整抗生素为哌拉西林他唑巴坦",
            "继续目前方案，加强康复训练",
            "增加氯氮平血药浓度监测频率",
        ]),
        "{day}": str(random.randint(1, 14)),
        "{nrs}": str(random.randint(1, 6)),
        "{drain}": str(random.randint(10, 80)),
    }
    result = template
    for k, v in replacements.items():
        result = result.replace(k, v)
    return result


# === 主生成逻辑 ===

CSV_FIELDS = [
    "gender", "age", "diagnosis_name", "chief_complaint", "present_illness",
    "past_history", "personal_history", "is_smoking", "smoking_duration",
    "family_history", "allergic_history", "department", "height", "weight",
    "disability_category", "disability_level", "assess_type_name",
    "assess_result_name", "assess_score", "assess_time",
    "progress_note", "progress_note_time",
    "name", "id_card_no", "registered_address",
    "disability_cert_no", "medical_insurance_no",
]


def generate_record(index: int, is_image_case: bool = False, severity: str = "L3") -> dict:
    """生成单条医疗记录。"""
    diag_name, dept, chief = random.choice(DIAGNOSES)
    gender = random.choice(["男", "女"])
    age = random.randint(18, 85)
    is_smoking = random.choice(["是", "否"])

    # 根据严重度选择现病史
    if severity == "L5":
        present = random.choice(PRESENT_ILLNESS_L5)
    elif severity == "L4":
        present = random.choice(PRESENT_ILLNESS_L4)
    else:
        present = random.choice(PRESENT_ILLNESS_L3)

    # 图片病例追加实际图片路径与 PACS 标注
    if is_image_case:
        img_paths = [
            "data/samples/syphilis_case.png",
            "data/samples/hiv_report.jpg",
            "data/samples/blood_routine.jpg",
            "data/samples/xray_chest.png",
        ]
        chosen_img = random.choice(img_paths)
        present += f" 【附PACS病例图片资料：{chosen_img}】"

    past = random.choice(PAST_HISTORIES)
    if is_smoking == "是":
        personal = random.choice(PERSONAL_HISTORY_SMOKERS)
        smoking_dur = f"{random.randint(5, 40)}年"
    else:
        personal = random.choice(PERSONAL_HISTORY_NON_SMOKERS)
        smoking_dur = ""

    family = random.choice(FAMILY_HISTORIES)
    allergy = random.choice(ALLERGIC_HISTORIES)

    height = random.randint(148, 190)
    weight = random.randint(40, 110)

    dis_cat, dis_lev, assess_type, assess_result, assess_score = random.choice(DISABILITY_CATEGORIES)
    assess_score_int = int(assess_score) + random.randint(-5, 5)
    assess_score_int = max(0, min(100, assess_score_int))

    progress = _fill_progress_note(random.choice(PROGRESS_NOTE_TEMPLATES), chief)

    return {
        "gender": gender,
        "age": str(age),
        "diagnosis_name": diag_name,
        "chief_complaint": chief,
        "present_illness": present,
        "past_history": past,
        "personal_history": personal,
        "is_smoking": is_smoking,
        "smoking_duration": smoking_dur,
        "family_history": family,
        "allergic_history": allergy,
        "department": dept,
        "height": str(height),
        "weight": str(weight),
        "disability_category": dis_cat,
        "disability_level": dis_lev,
        "assess_type_name": assess_type,
        "assess_result_name": assess_result,
        "assess_score": str(assess_score_int),
        "assess_time": gen_date_recent(),
        "progress_note": progress,
        "progress_note_time": gen_datetime_recent(),
        "name": gen_name(),
        "id_card_no": gen_id_card(),
        "registered_address": gen_address(),
        "disability_cert_no": gen_disability_cert(),
        "medical_insurance_no": gen_medical_insurance(),
    }


def generate_dataset(count: int = 20) -> list[dict]:
    """生成完整数据集。"""
    records = []
    # 确定图片病例索引（3-4条）
    image_indices = set(random.sample(range(count), min(4, count)))

    # 确定严重度分布：增加 L4/L5 病例密度（L4 30%, L5 20%）
    severity_map = {}
    l4_cnt = max(1, int(count * 0.3))
    l5_cnt = max(1, int(count * 0.2))

    l4_indices = set(random.sample(range(count), l4_cnt))
    remaining = set(range(count)) - l4_indices
    l5_indices = set(random.sample(sorted(remaining), min(l5_cnt, len(remaining))))

    for i in range(count):
        if i in l5_indices:
            severity_map[i] = "L5"
        elif i in l4_indices:
            severity_map[i] = "L4"
        else:
            severity_map[i] = "L3"

    for i in range(count):
        rec = generate_record(
            i,
            is_image_case=(i in image_indices),
            severity=severity_map[i],
        )
        records.append(rec)
    return records


def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="生成模拟医疗数据 CSV")
    parser.add_argument("--output", default=str(project_root / "data/kangyang.csv"), help="输出文件路径")
    parser.add_argument("--count", type=int, default=20, help="生成记录数")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = generate_dataset(args.count)

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    print(f"✅ 已生成 {len(records)} 条医疗记录 → {output_path.resolve()}")
    print(f"   字段数: {len(CSV_FIELDS)}")
    print(f"   图片病例: {sum(1 for r in records if 'data/samples/' in r['present_illness'])} 条")


if __name__ == "__main__":
    main()

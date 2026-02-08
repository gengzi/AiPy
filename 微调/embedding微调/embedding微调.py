from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import math
from sentence_transformers.evaluation import InformationRetrievalEvaluator
# === 1. 加载模型 ===
model_name = "/mnt/f/model/Qwen3-Embedding-0.6B"
## trust_remote_code 允许其执行可能包含的自定义模块或处理逻辑
model = SentenceTransformer(model_name, trust_remote_code=True, device="cuda")


# === 2. 设置超参数 ===
batch_size = 16  # 根据显存调整
num_epochs = 3
learning_rate = 3e-5
warmup_ratio = 0.1
max_seq_length = 512

# 设置截断/填充长度
model.max_seq_length = max_seq_length

your_data_list = [
  {
    "question": "“给定资料 2~3”反映了在反腐倡廉的背景下，对于改变过去工作生活中某些错误的习惯性做法，一些单位和个人态度不同、表现不一。请用精炼的语言依次将这些态度或表现归纳为若干类型。",
    "positive": "材料2描述了反腐背景下单位取消职工节日福利（如月饼）引发的争议，部分领导因怕“撞枪口”而“多一事不如少一事”；也有领导认为反腐与正常福利应区分对待。材料3指出，尽管中央明令禁止在风景名胜区开会，仍有单位通过“打擦边球”方式变相旅游开会，如设分会场、改会议名称、借中转之名停留景区等。这些态度或表现可归纳为：①矫枉过正，保守自保型；②坚决拥护且实事求是型；③上有政策下有对策的擦边球型。"
  },
  {
    "question": "请根据“给定资料 1”，概括“创客运动”在中国迅猛发展所产生的积极效果。",
    "positive": "材料1介绍了创客的定义、发展现状及其对制造业、教育、就业的影响。指出创客推动个性化定制、助力山寨企业转型、激活高校创新资源，并在人工智能替代人工背景下创造新就业岗位。创客运动的积极效果包括：一是推进以人为本的大众创新；二是引领制造业转型；三是谋划山寨出路；四是提升素质教育；五是创造就业岗位。"
  },
  {
    "question": "某市╳╳区政府网站开辟“新变化新风尚”专栏，请根据“给定资料 5”整理一篇新闻稿发表在该专栏。",
    "positive": "材料5通过小茜（网络挂号购票）、贾先生（电脑派位入学）、小林（请搬家公司）、小辉（网购电视）等案例，展现互联网普及和制度革新如何改变人们“托关系、靠人情”的旧习惯，转向高效、透明、自助的新生活方式。这些新变化包括：互联网发展使购票挂号更便捷；教育“新政”实现公平入学；搬家服务专业化替代亲友帮忙；电商价格透明无需“找关系”。靠科技、靠制度办事已蔚然成风，成为社区新风尚。"
  },
  {
    "question": "随着社会发展进步，很多“习惯”发生了改变。请你以“习惯的是是非非”为题，撰写一篇不少于800字的文章。",
    "positive": "全套材料共同构成背景：材料2-4反映反腐改变公款消费与福利发放习惯；材料5展示技术与制度推动生活办事方式革新；材料6-7揭示环保习惯（如垃圾分类、秸秆焚烧）难以改变的深层原因——缺乏配套支持与经济激励。文章应指出：习惯有好坏之分，老旧习惯（如公款吃喝、托关系、焚烧秸秆）源于历史条件，但已不适应新时代；需以创新思维推动变革，如制度约束（八项规定）、技术赋能（互联网）、产业配套（秸秆回收）；改变习惯非一日之功，需系统施策，扬弃“非黑即白”思维，做到“穷则变，变则通，通则久”。"
  },
{
    "question": "请根据‘给定资料2’概括信用体系建设对当前社会具有的积极影响。",
    "positive": "党的十八大以来，国家加快社会信用体系建设，初步形成守信联合激励和失信联合惩戒机制，基本建成统一的信用信息共享平台，“一处失信、处处受限”的格局逐步呈现。在法院执行领域，254万失信被执行人主动履行义务；在税收领域，1170户黑名单当事人缴清税款后被撤出；在电商领域，失信企业信息被推送至地方治理。信用体系提升了企业交易效率、降低风险、激发创新，方便民众在医疗教育等领域获得便利，倒逼个人守规，促进社会和谐，同时提高政府治理效率，推动政务精细化。"
  },
  {
    "question": "‘给定资料3’介绍了龙台村开展信用村创建的具体做法，请归纳其主要经验，供G县在全县范围内开展信用村创建工作时借鉴。",
    "positive": "龙台村以农业为主，村民曾因无抵押、无信用记录难以获贷。当地政府与银行成立创建信用村领导小组，制定实施方案，建立党群服务中心，政府、银行、村民三方签订守信合约，为村民建立永久信用档案并评级。对贫困户“一户一策”精准帮扶。通过宣传引导、正向激励，发展信用户并授信2-5万元低息贷款；信用户带动一般农户争创信用。同时结合产业特色推出“药农贷”、红薯种植扶贫贷款等定制化信贷产品，实现信贷+扶贫融合。主要经验包括：强化组织领导、三方签约建档案、信用评级授信、正向激励带动、产业精准对接。"
  },
  {
    "question": "根据‘给定资料6’，假定你是某高校团委负责人，请给学校党委写一份‘加大信用教育力度’的建议书。",
    "positive": "当前大学生信用认知严重不足：近六成缺乏信用知识，逾八成不了解个人信用报告。校园金融欺诈频发，部分学生将征信当作交易筹码，对欠贷后果漠然。团中央等多部门已联合开展“金融安全专家校园行”活动，通过讲座、路演普及信用安全知识，学生反馈收获大。建议学校党委将信用教育纳入思政体系，常态化开展专家进校园活动；开设信用知识选修课或专题培训；利用团组织网络加强宣传；建立校园信用约束与激励机制。此举有助于防范金融风险，培养具法治意识与契约精神的现代公民。"
  },
  {
    "question": "假设你是某市人社部门的工作人员，需要进社区向群众推广电子社保卡。请结合给定资料，草拟一份介绍电子社保卡的宣传稿。",
    "positive": "电子社保卡是实体社保卡的线上形态，与实体卡一一对应，是线上应用的有效凭证。它依托全国社保卡服务平台，通过手机即可申领，操作简便。目前全国持卡人数超12亿。电子社保卡已在多地实现“一码通城”：南宁可用于出入公园、图书馆；宁夏、潮州等地支持药店扫码购药；青岛推出“电子时间银行”记录志愿服务；多地医院支持线上缴费。它采用人脸识别、密码验证等多重安全技术，确保“实人实名实卡”，防止盗刷泄露。未来还将支持社保查询、资格认证、参保缴费等功能，并嵌入政务服务、智慧城市、金融等场景，让群众尽享信息时代的“速度与温情”。"
  },
  {
    "question": "根据给定资料1，概括改革开放以来，赵家村发生的变化。",
    "positive": "改革开放以来，赵家村发生翻天覆地变化：一是解决温饱，实行包产到户后粮食充足，白面肉食成家常；二是住房条件改善，院落硬化、通自来水、装玻璃窗、建果库，房屋整齐如小区；三是思想观念转变，从种粮转向种苹果等经济作物，收入大增，扩大规模、购置农机、流转土地；四是基础设施完善，通班车、建图书室；五是精神生活丰富，手机微信普及，村民结团旅游，休闲活动多样。"
  },
  {
    "question": "给定资料2中提到‘40年来经济的发展，在点滴的民生改善中找到了生动而深刻的注脚。’请谈谈你对这句话的理解。",
    "positive": "这句话意指经济发展成果最终体现在民生持续改善上。具体表现为：农村通过家庭联产承包、土地确权、三权分置、资源变资产等改革，促进农民增收致富；医疗方面，城乡居民医保和大病保险完善，实现花钱少、报销快、服务优，乡医老洪三次住院无负担；居住方面，棚户区改造通过货币化安置改善住房条件。这些“点滴”变化增强了人民获得感、幸福感，印证了发展为了人民的根本宗旨。"
  },
  {
    "question": "给定资料3提到了F县在2013年发展苗木产业时遇到的困难，请谈谈F县是如何破解这一难题的。",
    "positive": "F县苗木产业曾面临品种杂乱、质量低劣、供大于求、行情下滑、收入下降等问题，且产业结构单一、风险高。破解对策包括：一是升级产业，对接市场需求，种植高档精品苗木，推进专业化、标准化、品牌化；引进龙头企业，带来先进技术和市场理念，形成工厂化育苗模式。二是促进融合转型，建设精品苗木基地和国家生态公园，发展林苗结合；利用生态资源发展旅游，举办活动吸引游客，提升苗木附加值，打造“绿色银行”。"
  },
  {
    "question": "H县在垃圾分类上取得的成功，已成为该县一张新名片……请结合材料，分析H县做法的启示。",
    "positive": "H县在垃圾分类方面成效显著，被国际组织誉为“中国小城市村居垃圾回收利用和环境保护的典范”。其成功经验在于建立了适合农村实际的垃圾回收利用体系，注重源头分类、资源化利用与环境保护相结合，形成了可复制、可持续的治理模式，为全国小城市和乡村提供了示范样板。"
  },
  {
    "question": "材料四描述了Z市农村厕所问题及其改造，请概括农村改厕的意义及实施难点。",
    "positive": "Z市农村长期使用连茅圈、旱厕，不仅气味大、冬天寒冷，还污染地下水，成为蚊蝇孳生地和病菌传播源，80%以上传染病与此相关，且影响年轻人返乡。2014年起，Z市将改厕纳入美丽乡村建设，有效改善了人居环境、提升了村民健康水平、增强了乡村吸引力。但厕所虽小，改造却是系统工程，牵涉资金投入、技术适配、村民习惯改变、后续维护等多方面，需统筹规划、因地制宜、久久为功。"
  },
  {
    "question": "材料三中，老郑、郭书记和叶教授围绕乡村振兴谈了观察和思考。请归纳实现农业农村现代化的关键举措。",
    "positive": "实现农业农村现代化需多措并举：一是坚持农业农村优先发展，转变政绩观，资源向农村倾斜；二是加强基础设施建设，如修“农机路”、通电、完善水利；三是强化政策支持，如农机补贴、农技指导；四是推动产业融合，发展乡村旅游、创意农业、“科技+”“健康+”等新业态；五是补齐公共服务短板，推动城乡教育、养老、医疗一体化；六是吸引人才回流，鼓励农民工、高校毕业生返乡创业；七是科学规划村镇体系，保护传统村落，为乡村聚人气、添活力。"
  }
]

# === 数据划分 ===
# 将数据划分为训练集和测试集
train_data, test_data = train_test_split(your_data_list, test_size=0.2, random_state=3)
## 数据增强 （多语言，多问法）
print("训练集数据: ",train_data)
print("测试集数据: ",test_data)

# === 3. 准备数据 ===
# 假设你有一个 JSON 列表: [{"anchor": "...", "positive": "..."}, ...]
train_examples = [
    InputExample(texts=[item["question"], item["positive"]])
    for item in train_data
]



test_examples = [
    InputExample(texts=[item["question"], item["positive"]])
    for item in test_data
]
# === 3. 加载数据   训练集，测试集，评估集===
train_dataloader = DataLoader(
    train_examples,
    batch_size=batch_size,
    shuffle=True
)

test_dataloader = DataLoader(
    test_examples,
    batch_size=batch_size,
    shuffle=False
)

# === 4. 损失函数 ===
train_loss = losses.MultipleNegativesRankingLoss(model)

# === 5. 计算 warmup_steps ===
total_steps = len(train_dataloader) * num_epochs
warmup_steps = math.ceil(total_steps * warmup_ratio)

# 创建验证评估器，每个epoch结束后会在验证集上评估


# 准备数据
queries = {i: item["question"] for i, item in enumerate(test_data)}
corpus = {i: item["positive"] for i, item in enumerate(test_data)}
relevant_docs = {i: {i} for i in range(len(test_data))}  # 每个 query 的正例是自己

evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name='qa-retrieval',
    batch_size = 1
)

"""
model: SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: Qwen3Model 
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': True, 'include_prompt': True})
  (2): Normalize()
)
"""
print("model:",model)


# result = evaluator(model)
# print("评估结果：",result)
# === 6. 开始训练 ===
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    optimizer_params={"lr": learning_rate},
    warmup_steps=warmup_steps,
    output_path="/home/gengzi",

    show_progress_bar=True,
    use_amp=True,  # 启用混合精度加速（推荐）
    evaluation_steps=0,  # 0 = 只在 epoch 结束时评估
    save_best_model=True,  # 保存验证集上最好的模型（需 evaluator）
    checkpoint_path="/mnt/f/model/checkpoints",  # 可选：中间 checkpoint
    checkpoint_save_steps=500,  # 每 500 步存一次 checkpoint
    checkpoint_save_total_limit=2,  # 最多保留 2 个 checkpoint
)

print("\n" + "="*50)
print("🔍 训练结束后立即手动评估:")
results = evaluator(model)
for k, v in results.items():
    print(f"{k}: {v}")
print("="*50)
# === 7. 评估，计算loss ===

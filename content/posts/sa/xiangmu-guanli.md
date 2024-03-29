+++
title = '项目管理'
date = 2024-02-22T20:16:04+08:00
draft = true
tags = ["项目管理","系统分析师"]
categories = ["系统分析师"]
+++

## 项目管理

### 范围管理

- 规划范围管理（编制范围管理计划），
- 定义范围：编制项目范围说明书。输入包括：项目章程，项目范围管理计划，组织过程资产，批准的变更申请
- 创建工作分解结构（WBS），项目工作分解，自上而下
- 确认范围，确认可交付成功
- 范围控制，监督，管理变更

产品范围：功能，是否满足产品描述，是项目范围的基础
项目范围：项目必须做的工作，定义项目管理计划的基础，包括产品范围。项目范围说明书，WBS，WBS词典。

### 进度管理（时间管理）

活动：分解的工作包
活动定义、活动排序、活动资源估算、活动历时估算，进度计划编制，进度控制

软件规模估算魔像：cocomo（基本、中间、详细），cocomoII（应用组装模型，早期设计阶段模型，体系结构阶段模型）（不同估算选择：对象点，功能点，带马航）

#### 甘特图gantt

时间，并行关系

#### 项目计划评审技术（PERT）图

反应依赖关系

#### 关键路径法

最早开始时间（ES），最早结束（EF），最迟结束（LF），最迟开始（LS）
总浮动时间：关键活动总浮动时间为0
自由浮动时间
七格图。

### 成本

成本估算：自顶向下，自底向上，差别估算
成本预算：分配到工作包，应急储备（已知），管理储备（未知）
成本控制

成本类型：可变成本，固定成本，直接成本（团队独有），间接成本（多项目分摊），机会成本（失去的最大收益成本），沉没成本


### 软件配置管理

完整、可跟踪。配置项管理：版本、变更、基线。

制订配置管理计划、配置标识（识别）、配置控制、配置状态报告、配置审计、发布管理和交付。
<hr/>
配置项：主要属性 名称、标识符、文件状态、版本、作者、日期

基线配置项（设计文档、源程序），非基线配置项

状态：草稿（0.YZ），正式（X.Y），修改（X.YZ）

多次修改确定；任何修改产生新版本；按一定规则保存配置项所有版本，不能抛弃旧版；快速准确查找任何版本

基线：一组配置项，相对稳定。对应里程碑。发型基线（release，外部），构造基线（内部，build）

<hr/>
建立基线事件，受控的配置项，建立和变更基线的程序，批准变更的权限。

配置库：记录配置所有信息；评价变更后果；提取管理信息
- 开发库：动态库、程序员库、工作库。
- 受控库：基线+基线变更
- 产品库：静态库、发行库、软件仓库。已发布的存档。一般不修改

### 质量

范围、进度、成本影响质量。
质量规划（识别，描述过程），质量保证（质量审计，过程分析），质量控制（监控）

质量模型（**）
- 功能性：适合，准确，互操作，依从，安全
- 可靠性：成熟，容错，易恢复
- 可用性：易理解，易学，易操作
- 效率：时间，资源
- 可维护性：易分析，可修改，稳定（修改造成的结果），可测试
- 可移植性：适应，易安装，一致性，可替换

运行阶段，修正阶段，转移阶段

软件评审（质量评审）：设计的质量（设计规格说明书），程序的质量（程序执行）
容错技术：主要手段冗余。结构冗余（动态、静态、混合），信息冗余，时间冗余（重复执行）
冗余附加


### 风险

管理计划编制，风险识别，定性分析，定量分析（曝光度），应对计划编制，风险监控

项目风险：超过90%可提前应对和管理；今早识别；最有控制力一方承担风险；承担程度与回报匹配，有上限。
- 随机性，发生概率和后果偶然，遵循统计规律
- 相对性，收益，投入，级别高
- 可变性，条件变化引起风险变化

- 项目风险：预算、进度、个人、资源、用户、需求。威胁项目计划
- 技术风险：设计、实现、接口、测试、维护。威胁质量
- 商业风险：市场、策略、销售、管理、预算。威胁生存能力

## 补充

项目组织形式：项目型（项目经理绝对领导），职能型（部门领导为主），矩阵型（二者结合）

程序设计小组组织：
- 主程序员制（主程序员全权负责，后援能替代，适合大规模）
- 民主制（无主程序员），适用于规模小
- 层次式。组长-高级程序员-若干程序员
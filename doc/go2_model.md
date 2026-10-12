# `robotdog/go2_model.py` — MJCF 装配与场景编译

对应代码：`robotdog/go2_model.py`。相关坑见 `experience.md` §1、§8；本文只讲抽象结构。

## 职责

把 vendored 的 Unitree Go2 MJCF（**永不修改**）与自定义地形组装成一个可编译的
MuJoCo 场景，并从编译后的模型里提取环境需要的全部元信息。

## 核心抽象

### 文本装配，而不是 `<include>`

`build_scene_xml` 对 go2.xml 做**文本级**注入（正则定位 `<option>`、`<asset>`、
`</worldbody>` 三个点），再整体编译。不用 `<include>` 是刻意的：MuJoCo 把相对
资源路径解析到主文件目录，而从字符串编译时没有主文件目录。

### 网格走内存，不碰文件系统

本工作区路径含中文，MuJoCo 的窄字符文件读取会失败。解法是
`MjModel.from_xml_string(xml, assets=go2_assets())`：`go2_assets()` 把全部网格
读进内存按裸 basename 索引（`lru_cache` 缓存，每进程只付一次），`meshdir` 保留
原本的相对路径但永远不会发生目录查找。

### extract_meta / Go2Meta：编译产物的只读视图

从编译后的模型解析出全部 id/索引（关节、执行器、足端、力矩范围、home 姿态、
地形与碰撞 geom 分组）。两条不变量：

- **找不到名字就当场炸**（`_resolve_id` 抛 KeyError；`floor` 缺失抛 RuntimeError）
  —— 静默的 `-1` 会被当成「最后一个 geom」混进地形集合，接触判定全错。
- **Go2Meta 只保留有读者的字段**。躯干/腿碰撞体的拆分规则：`geom_group == 3` 是
  Go2 MJCF 里「所有会碰撞的图元」，焊在 base body 上的归躯干（真摔），其余
  （减去四个脚球）归腿（蹭立面，扣分不终止）。

### 帧约定工具

`quat_to_mat` / `quat_to_rpy` 是仅有的四元数换算实现（避免引 scipy）。
`home_qpos(meta, x=, y=, z=, yaw=)` 生成带平移/朝向的初始姿态。

## 踩坑指引

已知但故意不改的两处（home keyframe 注释与实际 XML 的形态差异、`_OPTION` 正则
只认自闭合标签）—— 见 `experience.md` §8，**不要顺手"修"**。

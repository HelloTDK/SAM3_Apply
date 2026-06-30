# SAM3 FastAPI AI图像分割服务 全知识点汇总文档
## 项目简介
基于FastAPI + SAM/Ultralytics 实现的图像检测、分割、相似目标检索推理服务；
支持Base64 JSON调用、前端文件上传、远程URL批量异步任务、两级API密钥鉴权、CUDA显存自动管理、并发限流。

# 一、Python 基础核心知识点
## 1. 类型提示 typing
- `Any`：任意不确定数据类型
- `Dict[K, V]` / `List[T]`：泛型容器注解
- `Optional[T]`：参数允许为None
- 函数返回值注解 `-> Dict[str, Any]`
- 复杂嵌套结构体 `List[Dict[str, Any]]` 存储图像样本元数据

## 2. 标准库应用
1. html
   - `html.escape()` HTML特殊字符转义，防御XSS注入，用于自定义API列表页面渲染
2. io.BytesIO
   - 内存二进制流，无需落地磁盘直接解析图片二进制数据
3. json
   - `json.loads()` 解析前端表单传递的JSON字符串（多样本接口sample_meta）
4. time
   - `time.monotonic()` 单调时钟，不受系统时间修改影响，计算模型空闲时长
5. traceback
   - `traceback.print_exc()` 打印完整异常堆栈，快速定位AI推理内部报错
6. functools.partial
   - 偏函数固化固定参数，简化多参数推理函数丢入线程池调用

## 3. PIL 图像处理
- `Image.open()` 二进制字节流加载图片
- `.convert("RGB")` 统一通道，规避RGBA透明通道推理报错
- `.copy()` 图像深拷贝，多样本复用原图不互相覆盖
- 工具封装：Base64解码、上传图片本地持久化保存

## 4. 异常处理规范
1. 分层捕获异常：
   - `ValueError` 参数格式错误，返回 HTTP 400
   - 通用Exception捕获推理未知异常，返回 HTTP 500
2. `raise HTTPException()` 主动中断请求并返回标准化错误
3. `from exc` 保留原始异常堆栈链路，便于排查
4. 单独捕获HTTPException，不重复包装错误信息

## 5. 数据结构与内置语法
- 列表推导式过滤多文件上传 `[item for item in form.getlist() if ...]`
- set集合去重，保存样本图片时避免重复写入
- `getattr()` 动态读取对象属性（路由解析、模型加载状态判断）
- `enumerate(start=1)` 带序号遍历，友好的数组下标错误提示
- 字典模拟复杂结构体：样本参考图、框坐标、正负样本标签统一结构

## 6. asyncio 并发同步原语
1. 信号量 `async with INFERENCE_SEMAPHORE`：限制最大并发推理数量
2. 线程锁 `INFERENCE_STATE_LOCK`：同步读写全局推理状态计数器
3. 全局变量 `ACTIVE_INFERENCE_COUNT` 实时统计运行中推理请求

## 7. 模块化分包设计
- 相对导入分包拆分职责：auth鉴权、config常量、image_utils图像工具、runtime模型推理、schemas请求模型、url_tasks异步任务
- 单一职责函数 `register_routes(app)` 统一注册全部接口，解耦应用启动与路由代码

# 二、FastAPI 全套核心知识点
## 1. 应用与路由基础
1. `FastAPI()` 应用主实例
2. `APIRoute` 底层路由对象，遍历`app.routes`实现自动扫描全部接口
3. 多URL绑定同一个接口函数：多个@get装饰器共用处理函数
4. 路由自动排序：按请求路径、请求方法字典序整理API列表

## 2. 四种请求参数接收方式
### （1）JSON 请求体（标准服务对接 /v1/*）
- Pydantic模型自动解析application/json请求体
- 内置参数类型校验，参数错误自动返回422

### （2）Multipart 表单文件上传（前端页面 /detect）
- `UploadFile = File(...)` 接收二进制图片文件
- `Form()` 接收表单字符串、数字参数
- `Request` 对象读取完整表单：`await request.form()`、`form.getlist()` 批量多文件上传

### （3）URL 路径参数
- `{task_id}` 路径变量，自动注入函数参数，用于任务查询、删除接口

### （4）URL 查询参数
- 接口尾部`?offset=0&limit=50`分页参数，自动解析int/float类型

## 3. 依赖注入 Depends 鉴权拦截
- `Depends(require_api_key)` 普通业务接口鉴权
- `Depends(require_admin_key)` 管理员密钥鉴权，仅允许操作密钥管理
- 自动解析路由依赖函数，区分接口鉴权等级（public/api/admin）

## 4. 文件上传 UploadFile 能力
1. 单文件必填校验 `File(...)`
2. 批量多同名文件 `form.getlist("key")`
3. 二进制异步读取 `await file.read()`
4. 文件体积上限校验，防止超大图片OOM

## 5. 响应返回类型
1. 默认JSON：`Dict[str, Any]` 字典序列化返回
2. `HTMLResponse`：返回原生HTML静态页面（首页、自定义API文档页）
3. 本地静态文件读取：通过配置STATIC_DIR读取index.html作为首页

## 6. 异步 async/await 工程实践
1. 全部接口使用`async def`异步定义，适配FastAPI事件循环
2. IO操作全部await：读文件、读取表单、等待线程推理、等待异步任务
3. 同步阻塞AI推理放入线程池`run_inference_in_thread`，不阻塞web服务事件循环

## 7. 自动接口文档体系
1. `/docs` Swagger交互式调试页面
2. `/openapi.json` 标准OpenAPI接口描述文件，可导入Postman
3. `/api-list.json` 自定义结构化接口清单，供外部系统同步接口元数据
4. 自定义可视化HTML接口文档页面 `/api-list` / `/apis`

## 8. 健康检查监控接口 /health
标准化服务监控端点，输出指标：
- 模型加载状态、当前活跃推理并发数
- GPU设备、显存占用统计
- 模型空闲时长、自动卸载开关、并发上限配置
- 推理后端、输入分辨率等运行配置

# 三、HTTP 网络通信知识点
## 1. HTTP 方法语义规范
- GET：查询类操作（无副作用，无请求体）：健康检查、任务查询、文档页面
- POST：创建/计算操作：推理请求、创建异步任务、生成API密钥、上传图片
- DELETE：资源销毁操作：取消任务、删除API密钥
- 过滤HEAD/OPTIONS预检请求，不展示在API文档列表

## 2. 两种主流请求体编码
1. application/json：后端微服务对接，图片用Base64字符串传输
2. multipart/form-data：前端网页上传，二进制图片+表单参数混合传输

## 3. 两种图像传输方案对比
1. Base64 JSON传输：适合服务间调用，无需文件存储；大图会增大请求包体积
2. Multipart文件上传：前端友好，二进制压缩，批量图片传输带宽占用更低

## 4. Web安全防护
XSS防御：HTML页面渲染时全部字段使用`html.escape()`转义，防止注入恶意JS脚本

## 5. 长耗时异步任务设计模式
1. 创建任务POST /tasks 返回唯一task_id
2. GET /tasks/{task_id} 查询任务整体状态
3. GET /tasks/{task_id}/results 分页获取批量推理结果，支持等待超时
4. DELETE /tasks/{task_id} 主动终止未完成任务
核心：TASK_REGISTRY 全局任务注册表，解耦即时推理与批量长任务

# 四、AI图像推理 & CUDA GPU 工程知识点
## 1. 全局单例模型架构
- predictor全局单例加载SAM/Ultralytics大模型，避免重复加载权重消耗显存
- MODEL_LOCK串行锁，多请求互斥访问模型，防止图片/特征缓存互相覆盖

## 2. 三层并发限流机制
1. 信号量INFERENCE_SEMAPHORE：限制全局最大并发推理MAX_CONCURRENT_INFERENCES
2. 模型串行锁SERIALIZE_MODEL_ACCESS：多请求互斥使用模型
3. 全局计数器ACTIVE_INFERENCE_COUNT：健康接口实时暴露并发指标

## 3. GPU显存自动回收策略
1. IDLE_MODEL_UNLOAD_SECONDS：模型空闲超时自动卸载释放显存
2. CUDA_CLEANUP_AFTER_REQUEST：单次推理完成清空CUDA缓存
3. 健康接口输出显存占用，监控显存泄漏问题
4. IDLE_MODEL_UNLOADED标记记录模型卸载状态

## 4. 多场景推理流水线封装
1. run_detection_pipeline：文本prompt通用目标检测分割
2. run_box_segmentation_pipeline：指定框坐标精准分割
3. run_similar_object_batch_pipeline：单参考图相似目标检索
4. run_multi_similar_object_batch_pipeline：多正负样本批量相似检索
5. run_by_url_request：远程网络图片下载后推理，适配批量URL任务

## 5. 图像推理可调参数
- confidence_threshold：检测置信度过滤
- similarity_threshold：特征相似度阈值
- sam_threshold：SAM掩码生成阈值
- nms_iou：非极大抑制重叠阈值
- polygon_simplify_epsilon：轮廓多边形简化系数，控制返回坐标点数

# 五、API 鉴权权限系统知识点
## 1. 两级权限隔离
1. 普通API Key（require_api_key）：仅允许调用推理、任务查询业务接口
2. Admin管理员Key（require_admin_key）：拥有密钥完整管理权限（增删查）

## 2. 两套密钥管理接口
1. /ui/api-keys：无鉴权简易接口，内部后台管理页面使用
2. /v1/api-keys：管理员鉴权标准接口，自动化程序批量管理密钥

## 3. 密钥管理能力
- 创建密钥：自定义名称、角色、过期天数
- 列出全部密钥元数据（不返回完整明文密钥，仅创建时展示一次）
- 删除密钥，内置保护逻辑：禁止删除最后一个管理员密钥

# 六、项目工程化架构设计知识点
## 1. 分层解耦分层职责
1. routes（当前代码）：仅处理HTTP接收、参数校验、调用底层推理函数
2. schemas：Pydantic请求/响应结构体定义，统一入参出参规范
3. auth：API密钥鉴权校验逻辑
4. config：全局常量集中配置，消除硬编码
5. image_utils：图片解码、存储通用工具函数
6. runtime：模型加载、并发锁、推理核心流水线
7. url_tasks：异步批量任务注册表、任务等待/分页逻辑

## 2. 代码复用抽象设计
1. 全部推理接口统一并发模板，代码结构高度统一便于维护
2. build_single_positive_sample 统一构造样本结构体，消除重复字典赋值代码
3. 自动扫描路由函数，无需手动维护API文档清单

## 3. 可观测、可调试设计
1. /health 健康接口暴露全量运行指标，监控并发、显存、模型状态
2. 所有异常打印完整堆栈，快速定位推理内部错误
3. SAVE_UPLOADS开关：保存所有输入原图，线上问题复现调试
4. pic_id业务追踪ID：每个推理请求携带业务标识，日志链路溯源

## 4. 健壮容错校验
1. 图片体积上限校验，防止超大图OOM
2. 图片格式、Base64合法性校验
3. JSON参数格式、数组下标越界校验（多样本文件索引）
4. 资源不存在统一返回404（任务、密钥）
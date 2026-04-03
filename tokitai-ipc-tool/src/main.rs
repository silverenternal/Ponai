//! 3D 点云车机应用 - 主程序示例
//!
//! 演示如何使用 AI 调度层、点云工具层和实例分割工具层

use lidar_ai_studio::{
    AiScheduler, AiSchedulerConfig, InstanceSegToolManager, LidarAiError, PointCloudToolManager,
};
use tokitai::ToolProvider;

#[tokio::main]
async fn main() -> Result<(), LidarAiError> {
    // 初始化日志
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("lidar_ai_studio=info".parse().unwrap()),
        )
        .init();

    println!("=== 3D 点云车机应用 - AI 调度框架 ===\n");

    // ==================== 1. 初始化点云工具层 ====================
    println!("1. 初始化点云工具层...");
    let tools = PointCloudToolManager::new_python("python_tools/pointcloud_tools.py")?;
    println!("   ✓ Python 点云工具已加载\n");

    // ==================== 2. 获取工具定义 ====================
    println!("2. 获取工具定义（用于 AI 调用）...");
    let tool_definitions = PointCloudToolManager::tool_definitions();
    println!("   可用工具：");
    for tool in tool_definitions {
        println!("   - {}: {}", tool.name, tool.description);
    }
    println!();

    // ==================== 3. 初始化 AI 调度器 ====================
    println!("3. 初始化 AI 调度器（Ollama 适配）...");
    let config = AiSchedulerConfig {
        host: "localhost".to_string(),
        port: 11434,
        model: "llama3.2".to_string(),
        stream: false,
    };
    let scheduler = AiScheduler::new(config);

    // 注册工具到 AI 调度器
    let ollama_tools: Vec<_> = tool_definitions
        .iter()
        .map(|t| lidar_ai_studio::ai_scheduler::ToolDefinition {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: serde_json::from_str(&t.input_schema).unwrap_or_default(),
        })
        .collect();
    scheduler.register_tools(ollama_tools).await;
    println!("   ✓ AI 调度器已配置（Ollama: localhost:11434）");
    println!("   ✓ 工具已注册到 AI 调度器\n");

    // ==================== 4. 演示点云工具调用 ====================
    println!("4. 演示点云工具调用...");

    // 加载点云
    match tools.load_point_cloud("/data/pointcloud.pcd".to_string()) {
        Ok(result) => println!("   加载点云：{}", result),
        Err(e) => println!("   加载点云（模拟）：{}", e),
    }

    // 降采样
    match tools.downsample(0.05) {
        Ok(result) => println!("   降采样：{}", result),
        Err(e) => println!("   降采样（模拟）：{}", e),
    }

    println!();

    // ==================== 5. 实例分割工具演示 ====================
    println!("5. 实例分割工具演示...");

    // 创建实例分割工具管理器（IPC 模式）
    let mut seg_tools = InstanceSegToolManager::new_ipc("python_tools/instance_seg_tools.py")?;
    println!("   ✓ 实例分割工具已加载（IPC 模式）");

    // 获取后端信息
    match seg_tools.get_backend_info() {
        Ok(info) => println!("   后端信息：{}", info),
        Err(e) => println!("   后端信息：{}", e),
    }

    // 演示切换到 HTTP 模式
    println!("\n   演示切换到 HTTP 后端...");
    seg_tools.switch_to_http("http://localhost:8080", None, 30);
    println!("   ✓ 当前后端：{}", seg_tools.current_backend());

    // 切换回 IPC
    println!("\n   切换回 IPC 后端...");
    seg_tools.switch_to_ipc("python_tools/instance_seg_tools.py")?;
    println!("   ✓ 当前后端：{}", seg_tools.current_backend());

    // 演示动态切换 API
    println!("\n   使用 switch_backend API 切换...");
    match seg_tools.switch_backend("http".to_string(), Some("http://gpu-server:8080".to_string())) {
        Ok(msg) => println!("   {}", msg),
        Err(e) => println!("   切换失败：{}", e),
    }
    match seg_tools.switch_backend("ipc".to_string(), None) {
        Ok(msg) => println!("   {}", msg),
        Err(e) => println!("   切换失败：{}", e),
    }

    println!();

    // ==================== 6. AI 调度演示 ====================
    println!("6. AI 调度演示（需要 Ollama 服务）...");
    println!("   注意：此步骤需要本地运行 Ollama 服务");
    println!("   启动命令：ollama serve");
    println!();

    println!("   ✓ 框架已就绪，等待 Ollama 服务...");
    println!();

    // ==================== 7. 跨语言 IPC 说明 ====================
    println!("=== 跨语言 IPC/HTTP 双模式架构 ===");
    println!();
    println!("本框架支持两种后端调用模式，可通过 Switch 机制无缝切换：");
    println!();
    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│                  InstanceSegToolManager                 │");
    println!("│  ┌───────────────────────────────────────────────────┐  │");
    println!("│  │              BackendSwitch (切换器)               │  │");
    println!("│  │  ┌─────────────┐      ┌─────────────┐            │  │");
    println!("│  │  │ IPC Backend │      │ HTTP Backend│            │  │");
    println!("│  │  │  (本地)     │      │  (网络)     │            │  │");
    println!("│  │  └─────────────┘      └─────────────┘            │  │");
    println!("│  └───────────────────────────────────────────────────┘  │");
    println!("└─────────────────────────────────────────────────────────┘");
    println!();
    println!("模式对比:");
    println!("  IPC 模式：低延迟，无需网络，适合本地开发");
    println!("  HTTP 模式：可远程部署到 GPU 服务器，支持负载均衡");
    println!();
    println!("启动 HTTP 服务：cd python_tools && ./start_server.sh");
    println!();

    Ok(())
}

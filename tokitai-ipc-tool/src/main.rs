//! 3D 点云车机应用 - 主程序示例
//!
//! 演示如何使用 AI 调度层和点云工具层

use lidar_ai_studio::{
    AiScheduler, AiSchedulerConfig, LidarAiError, PointCloudToolManager,
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

    // ==================== 4. 演示工具调用 ====================
    println!("4. 演示点云工具调用...");

    // 加载点云
    match tools.load_point_cloud("/data/pointcloud.pcd".to_string()) {
        Ok(result) => println!("   加载点云：{}", result),
        Err(e) => println!("   加载点云（模拟）：{}", e),
    }

    // 获取点云信息
    match tools.get_point_cloud_info() {
        Ok(result) => println!("   点云信息：{}", result),
        Err(e) => println!("   点云信息（模拟）：{}", e),
    }

    // 降采样
    match tools.downsample(0.05) {
        Ok(result) => println!("   降采样：{}", result),
        Err(e) => println!("   降采样（模拟）：{}", e),
    }

    // 法线估计
    match tools.estimate_normals(20) {
        Ok(result) => println!("   法线估计：{}", result),
        Err(e) => println!("   法线估计（模拟）：{}", e),
    }

    // 离群点移除
    match tools.remove_outliers(30, 2.0) {
        Ok(result) => println!("   离群点移除：{}", result),
        Err(e) => println!("   离群点移除（模拟）：{}", e),
    }

    // 平面分割
    match tools.segment_plane(0.2, 1000) {
        Ok(result) => println!("   平面分割：{}", result),
        Err(e) => println!("   平面分割（模拟）：{}", e),
    }

    // 欧式聚类
    match tools.euclidean_clustering(0.5, 100, 10000) {
        Ok(result) => println!("   欧式聚类：{}", result),
        Err(e) => println!("   欧式聚类（模拟）：{}", e),
    }

    // 保存点云
    match tools.save_point_cloud("/data/output.pcd".to_string()) {
        Ok(result) => println!("   保存点云：{}", result),
        Err(e) => println!("   保存点云（模拟）：{}", e),
    }

    println!();

    // ==================== 5. AI 调度演示 ====================
    println!("5. AI 调度演示（需要 Ollama 服务）...");
    println!("   注意：此步骤需要本地运行 Ollama 服务");
    println!("   启动命令：ollama serve");
    println!();

    // 以下是 AI 调度的示例代码（需要 Ollama 服务）
    //
    // let user_message = "帮我加载点云文件并进行降采样和平面分割";
    // println!("用户：{}", user_message);
    //
    // // AI 理解并决定调用哪些工具
    // let response = scheduler.chat(user_message).await?;
    //
    // if let Some(tool_calls) = response.tool_calls {
    //     for tool_call in tool_calls {
    //         println!("AI 请求调用工具：{}", tool_call.function.name);
    //
    //         // 执行工具调用
    //         let result = execute_tool_call(&tools, &tool_call).await?;
    //
    //         // 将结果返回给 AI
    //         let final_response = scheduler
    //             .process_tool_result(&tool_call, result)
    //             .await?;
    //
    //         println!("AI 响应：{}", final_response.content);
    //     }
    // }

    println!("   ✓ 框架已就绪，等待 Ollama 服务...");
    println!();

    // ==================== 6. 跨语言 IPC 说明 ====================
    println!("=== 跨语言 IPC 架构说明 ===");
    println!();
    println!("本框架通过 stdin/stdout JSON Lines 实现跨语言工具调用：");
    println!();
    println!("┌─────────────┐      JSON       ┌─────────────┐");
    println!("│   Rust      │ ◄─────────────► │   Python    │");
    println!("│  (tokitai)  │   stdin/stdout  │  (Open3D)   │");
    println!("└─────────────┘                 └─────────────┘");
    println!("      │                               │");
    println!("      ▼                               ▼");
    println!("┌─────────────┐                 ┌─────────────┐");
    println!("│  AI 调度层   │                 │  点云算法   │");
    println!("│  (Ollama)   │                 │  (待实现)   │");
    println!("└─────────────┘                 └─────────────┘");
    println!();
    println!("请求格式：{{\"tool\": \"tool_name\", \"args\": {{...}}}}");
    println!("响应格式：{{\"result\": {{...}}, \"error\": null}}");
    println!();

    Ok(())
}

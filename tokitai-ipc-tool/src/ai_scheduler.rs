//! AI 调度层 - Ollama 适配器
//!
//! 负责与本地 Ollama 服务通信，调度 AI 模型调用工具

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::{LidarAiError, Result};

/// Ollama 配置
#[derive(Debug, Clone)]
pub struct AiSchedulerConfig {
    /// Ollama 服务地址
    pub host: String,
    /// Ollama 服务端口
    pub port: u16,
    /// 使用的模型名称
    pub model: String,
    /// 是否启用流式响应
    pub stream: bool,
}

impl Default for AiSchedulerConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 11434,
            model: "llama3.2".to_string(),
            stream: false,
        }
    }
}

/// 工具定义（发送给 AI）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// 聊天消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// 工具调用请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

/// 函数调用
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: Value,
}

/// Ollama API 响应
#[derive(Debug, Deserialize)]
pub struct OllamaResponse {
    pub model: String,
    pub message: ChatMessage,
    pub done: bool,
}

/// AI 调度器 - 管理 Ollama 会话和工具调用
pub struct AiScheduler {
    config: AiSchedulerConfig,
    client: reqwest::Client,
    conversation_history: Arc<RwLock<Vec<ChatMessage>>>,
    tools: Arc<RwLock<Vec<ToolDefinition>>>,
}

impl AiScheduler {
    /// 创建新的 AI 调度器
    pub fn new(config: AiSchedulerConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
            conversation_history: Arc::new(RwLock::new(Vec::new())),
            tools: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// 注册工具定义
    pub async fn register_tools(&self, tools: Vec<ToolDefinition>) {
        let mut self_tools = self.tools.write().await;
        *self_tools = tools;
    }

    /// 获取已注册的工具
    pub async fn get_tools(&self) -> Vec<ToolDefinition> {
        self.tools.read().await.clone()
    }

    /// 转换 tokitai 工具定义为 Ollama 格式
    pub fn to_ollama_tools(tools: &[ToolDefinition]) -> Value {
        let tools_list: Vec<Value> = tools.iter().map(|t| {
            let schema: serde_json::Map<String, Value> = 
                serde_json::from_str(&t.input_schema.to_string()).unwrap_or_default();
            json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": schema
                }
            })
        }).collect();
        json!({ "tools": tools_list })
    }

    /// 发送消息给 Ollama
    pub async fn chat(&self, user_message: &str) -> Result<ChatMessage> {
        // 添加用户消息到历史
        {
            let mut history = self.conversation_history.write().await;
            history.push(ChatMessage {
                role: "user".to_string(),
                content: user_message.to_string(),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // 获取工具列表
        let tools = self.get_tools().await;
        let ollama_tools = Self::to_ollama_tools(&tools);

        // 构建请求
        let url = format!("http://{}:{}/api/chat", self.config.host, self.config.port);
        
        let request_body = json!({
            "model": self.config.model,
            "messages": *self.conversation_history.read().await,
            "tools": ollama_tools,
            "stream": self.config.stream
        });

        tracing::info!("发送请求到 Ollama: {}", url);
        tracing::debug!("请求体：{}", serde_json::to_string_pretty(&request_body)?);

        // 发送请求
        let response = self.client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LidarAiError::AiScheduler(format!("Ollama 请求失败：{}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LidarAiError::AiScheduler(
                format!("Ollama 返回错误 {}: {}", status, error_text)
            ));
        }

        let ollama_response: OllamaResponse = response.json().await
            .map_err(|e| LidarAiError::AiScheduler(format!("解析响应失败：{}", e)))?;

        // 保存 AI 响应到历史
        {
            let mut history = self.conversation_history.write().await;
            history.push(ollama_response.message.clone());
        }

        Ok(ollama_response.message)
    }

    /// 处理工具调用结果并继续对话
    pub async fn process_tool_result(
        &self,
        tool_call: &ToolCall,
        result: Value,
    ) -> Result<ChatMessage> {
        // 添加工具结果到历史
        {
            let mut history = self.conversation_history.write().await;
            history.push(ChatMessage {
                role: "tool".to_string(),
                content: result.to_string(),
                tool_calls: None,
                tool_call_id: Some(tool_call.id.clone()),
            });
        }

        // 获取工具列表
        let tools = self.get_tools().await;
        let ollama_tools = Self::to_ollama_tools(&tools);

        // 发送结果给 Ollama 获取最终响应
        let url = format!("http://{}:{}/api/chat", self.config.host, self.config.port);
        
        let request_body = json!({
            "model": self.config.model,
            "messages": *self.conversation_history.read().await,
            "tools": ollama_tools,
            "stream": self.config.stream
        });

        let response = self.client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LidarAiError::AiScheduler(format!("Ollama 请求失败：{}", e)))?;

        let ollama_response: OllamaResponse = response.json().await
            .map_err(|e| LidarAiError::AiScheduler(format!("解析响应失败：{}", e)))?;

        // 保存 AI 响应到历史
        {
            let mut history = self.conversation_history.write().await;
            history.push(ollama_response.message.clone());
        }

        Ok(ollama_response.message)
    }

    /// 清除会话历史
    pub async fn clear_history(&self) {
        let mut history = self.conversation_history.write().await;
        history.clear();
    }

    /// 获取会话历史
    pub async fn get_history(&self) -> Vec<ChatMessage> {
        self.conversation_history.read().await.clone()
    }
}

/// 工具调用 trait（用于 AI 调度）
#[async_trait]
pub trait ToolCaller: Send + Sync {
    async fn call_tool_async(&self, name: &str, args: &Value) -> Result<Value>;
}

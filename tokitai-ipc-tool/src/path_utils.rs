//! 路径工具 - 运行时路径解析
//!
//! # 设计说明
//!
//! `env!("CARGO_MANIFEST_DIR")` 在编译时展开，适用于：
//! - 二进制文件（路径相对于编译时的项目目录）
//! - 测试代码（路径相对于 `cargo test` 运行目录）
//!
//! 但对于以下场景需要运行时路径解析：
//! - 库用户调用（库被安装到系统目录）
//! - 二进制文件被移动到其他位置
//!
//! # 使用方法
//!
//! ```rust
//! use lidar_ai_studio::path_utils::PathResolver;
//!
//! // 获取相对于可执行文件的路径（或回退到当前工作目录）
//! match PathResolver::resolve_relative("Cargo.toml") {
//!     Some(path) => println!("Found: {:?}", path),
//!     None => println!("Path not found"),
//! }
//!
//! // 或者使用回退到当前工作目录的简化版本
//! let path = PathResolver::resolve_relative_cwd("Cargo.toml");
//! ```

use std::path::{Path, PathBuf};
use std::env;

/// 路径解析器
///
/// 提供多种路径解析策略，确保在不同部署场景下都能找到资源文件。
pub struct PathResolver;

impl PathResolver {
    /// 解析相对路径（优先相对于可执行文件，回退到当前工作目录）
    ///
    /// # 解析顺序
    ///
    /// 1. 相对于可执行文件所在目录
    /// 2. 相对于当前工作目录
    ///
    /// # 参数
    /// - `relative_path`: 相对路径（如 `"python_tools/tool.py"`）
    ///
    /// # 返回
    /// - `Some(PathBuf)`: 解析成功（路径存在）
    /// - `None`: 解析失败（路径不存在）
    ///
    /// # 示例
    ///
    /// ```rust,no_run
    /// use lidar_ai_studio::path_utils::PathResolver;
    ///
    /// let script = PathResolver::resolve_relative("python_tools/tool.py")
    ///     .expect("Script not found");
    /// ```
    pub fn resolve_relative<P: AsRef<Path>>(relative_path: P) -> Option<PathBuf> {
        let relative = relative_path.as_ref();
        
        // 策略 1: 相对于可执行文件
        if let Ok(exe_path) = env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                let candidate = exe_dir.join(relative);
                if candidate.exists() {
                    return Some(candidate);
                }
            }
        }
        
        // 策略 2: 相对于当前工作目录
        let cwd_path = env::current_dir().ok()?.join(relative);
        if cwd_path.exists() {
            return Some(cwd_path);
        }
        
        // 策略 3: 相对于 CARGO_MANIFEST_DIR（仅调试模式有效）
        if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
            let manifest_path = PathBuf::from(manifest_dir).join(relative);
            if manifest_path.exists() {
                return Some(manifest_path);
            }
        }
        
        None
    }

    /// 解析相对路径（仅相对于当前工作目录）
    ///
    /// # 参数
    /// - `relative_path`: 相对路径
    ///
    /// # 返回
    /// 解析后的绝对路径（不检查是否存在）
    pub fn resolve_relative_cwd<P: AsRef<Path>>(relative_path: P) -> PathBuf {
        env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(relative_path)
    }

    /// 解析相对于可执行文件的路径
    ///
    /// # 参数
    /// - `relative_path`: 相对路径
    ///
    /// # 返回
    /// 解析后的绝对路径（不检查是否存在）
    pub fn resolve_relative_exe<P: AsRef<Path>>(relative_path: P) -> PathBuf {
        match env::current_exe() {
            Ok(exe_path) => {
                exe_path.parent()
                    .map(|p| p.join(relative_path))
                    .unwrap_or_else(|| PathBuf::from("."))
            }
            Err(_) => PathBuf::from("."),
        }
    }

    /// 确保路径存在，不存在则返回错误
    ///
    /// # 参数
    /// - `path`: 路径
    ///
    /// # 返回
    /// - `Ok(PathBuf)`: 路径存在
    /// - `Err(String)`: 路径不存在
    pub fn ensure_exists<P: AsRef<Path>>(path: P) -> Result<PathBuf, String> {
        let path_buf = path.as_ref().to_path_buf();
        if path_buf.exists() {
            Ok(path_buf)
        } else {
            Err(format!("Path does not exist: {}", path_buf.display()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_relative_cwd() {
        // Cargo.toml 应该在项目根目录
        let path = PathResolver::resolve_relative_cwd("Cargo.toml");
        assert!(path.is_absolute());
        assert!(path.ends_with("Cargo.toml"));
    }

    #[test]
    fn test_resolve_relative_exe() {
        // 测试相对于可执行文件的路径解析
        let path = PathResolver::resolve_relative_exe("Cargo.toml");
        assert!(path.ends_with("Cargo.toml"));
    }

    #[test]
    fn test_ensure_exists_success() {
        let result = PathResolver::ensure_exists("Cargo.toml");
        assert!(result.is_ok());
    }

    #[test]
    fn test_ensure_exists_failure() {
        let result = PathResolver::ensure_exists("/nonexistent/path/file.txt");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not exist"));
    }
}

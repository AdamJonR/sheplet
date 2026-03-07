use crate::error::ParserError;
use crate::extract::detect_format;
use std::path::{Path, PathBuf};

/// Recursively walk a directory and return all supported files.
/// Skips hidden files/directories (starting with '.').
/// Results are sorted by path for deterministic ordering.
pub fn walk_directory(dir: &Path) -> Result<Vec<PathBuf>, ParserError> {
    let mut files = Vec::new();
    walk_recursive(dir, &mut files)?;
    files.sort();
    Ok(files)
}

fn walk_recursive(dir: &Path, files: &mut Vec<PathBuf>) -> Result<(), ParserError> {
    let entries = std::fs::read_dir(dir).map_err(|e| ParserError::WalkError {
        path: dir.display().to_string(),
        source: e,
    })?;

    for entry in entries {
        let entry = entry.map_err(|e| ParserError::WalkError {
            path: dir.display().to_string(),
            source: e,
        })?;

        let path = entry.path();
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();

        // Skip hidden files/directories
        if name.starts_with('.') {
            continue;
        }

        if path.is_dir() {
            walk_recursive(&path, files)?;
        } else if detect_format(&path).is_some() {
            files.push(path);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_walk_nonexistent_dir() {
        let result = walk_directory(Path::new("/nonexistent/directory"));
        assert!(result.is_err());
    }

    #[test]
    fn test_walk_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let result = walk_directory(dir.path()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_walk_with_supported_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("doc.txt"), "hello").unwrap();
        std::fs::write(dir.path().join("data.csv"), "a,b\n1,2").unwrap();
        std::fs::write(dir.path().join("ignored.xyz"), "nope").unwrap();

        let result = walk_directory(dir.path()).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_walk_skips_hidden_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("visible.txt"), "hello").unwrap();
        std::fs::write(dir.path().join(".hidden.txt"), "secret").unwrap();

        let result = walk_directory(dir.path()).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].file_name().unwrap().to_str().unwrap() == "visible.txt");
    }

    #[test]
    fn test_walk_skips_hidden_directories() {
        let dir = tempfile::tempdir().unwrap();
        let hidden_dir = dir.path().join(".hidden");
        std::fs::create_dir(&hidden_dir).unwrap();
        std::fs::write(hidden_dir.join("secret.txt"), "hidden content").unwrap();
        std::fs::write(dir.path().join("visible.txt"), "visible content").unwrap();

        let result = walk_directory(dir.path()).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_walk_recursive() {
        let dir = tempfile::tempdir().unwrap();
        let subdir = dir.path().join("subdir");
        std::fs::create_dir(&subdir).unwrap();
        std::fs::write(dir.path().join("top.txt"), "top").unwrap();
        std::fs::write(subdir.join("nested.txt"), "nested").unwrap();

        let result = walk_directory(dir.path()).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_walk_sorted_output() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("c.txt"), "c").unwrap();
        std::fs::write(dir.path().join("a.txt"), "a").unwrap();
        std::fs::write(dir.path().join("b.txt"), "b").unwrap();

        let result = walk_directory(dir.path()).unwrap();
        let names: Vec<&str> = result
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(names, vec!["a.txt", "b.txt", "c.txt"]);
    }
}

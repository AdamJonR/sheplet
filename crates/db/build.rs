fn main() {
    if std::env::var("PROTOC").is_err()
        && let Some(home) = std::env::var_os("HOME") {
            let home = std::path::PathBuf::from(home);
            let protoc_candidates = [
                home.join("miniconda3/envs/ml-env/bin/protoc"),
                home.join("miniconda3/bin/protoc"),
                home.join("anaconda3/bin/protoc"),
            ];
            for protoc in &protoc_candidates {
                if protoc.exists() {
                    // SAFETY: build scripts are single-threaded
                    unsafe { std::env::set_var("PROTOC", protoc) };
                    break;
                }
            }
        }
}

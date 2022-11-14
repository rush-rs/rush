use rush_analyzer::DiagnosticLevel;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

struct Backend {
    client: Client,
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                ..ServerCapabilities::default()
            },
            server_info: Some(ServerInfo {
                name: env!("CARGO_PKG_NAME").to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        })
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.create_diagnostics(TextDocumentItem {
            language_id: "rush".to_string(),
            uri: params.text_document.uri,
            text: params.text_document.text,
            version: params.text_document.version,
        })
        .await
    }

    async fn did_change(&self, mut params: DidChangeTextDocumentParams) {
        self.create_diagnostics(TextDocumentItem {
            language_id: "rush".to_string(),
            uri: params.text_document.uri,
            version: params.text_document.version,
            text: std::mem::take(&mut params.content_changes[0].text),
        })
        .await
    }
}

impl Backend {
    async fn create_diagnostics(&self, params: TextDocumentItem) {
        // obtain the diagnostics from the analyzer
        let raw_diagnostics = match rush_analyzer::analyze(&params.text, params.uri.as_str()) {
            Ok((_, diagnostics)) => diagnostics,
            Err(diagnostics) => diagnostics,
        };
        // transform the diagnostics into the LSP form
        let diagnostics = raw_diagnostics
            .iter()
            .map(|diagnostic| {
                Diagnostic::new(
                    Range::new(
                        Position::new(
                            (diagnostic.span.start.line - 1) as u32,
                            (diagnostic.span.start.column - 1) as u32,
                        ),
                        Position::new(
                            (diagnostic.span.end.line - 1) as u32,
                            (diagnostic.span.end.column - 1) as u32,
                        ),
                    ),
                    Some(match diagnostic.level {
                        DiagnosticLevel::Hint => DiagnosticSeverity::HINT,
                        DiagnosticLevel::Info => DiagnosticSeverity::INFORMATION,
                        DiagnosticLevel::Warning => DiagnosticSeverity::WARNING,
                        DiagnosticLevel::Error(_) => DiagnosticSeverity::ERROR,
                    }),
                    None,
                    Some("rush-analyzer".to_string()),
                    format!(
                        "{}{}",
                        if let DiagnosticLevel::Error(kind) = &diagnostic.level {
                            format!("{}: ", kind)
                        } else {
                            "".to_string()
                        },
                        diagnostic.message
                    ),
                    None,
                    None,
                )
            })
            .collect();

        self.client
            .publish_diagnostics(params.uri.clone(), diagnostics, Some(params.version))
            .await;
    }
}

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| Backend { client });
    Server::new(stdin, stdout, socket).serve(service).await;
}

; Unshipped analyzer release
; https://github.com/dotnet/roslyn-analyzers/blob/main/src/Microsoft.CodeAnalysis.Analyzers/ReleaseTrackingAnalyzers.Help.md

### New Rules

Rule ID | Category | Severity | Notes
--------|----------|----------|-------
MSG001 | SourceGeneration | Error | Module source generator error
MSG002 | SourceGeneration | Warning | Invalid module method format
MSG003 | SourceGeneration | Error | Initializer class must not be named 'Init'
MSG004 | SourceGeneration | Info | RNG streams of this module can be pinned
MSG005 | SourceGeneration | Warning | Unsupported loop variable assignment
MSG006 | SourceGeneration | Warning | Parameters cannot be named after their locals
MSG007 | SourceGeneration | Warning | Plain C# loop stacks layers

namespace Shorokoo.Core.Nodes
{
    /// <summary>
    /// The call stack a node was built from, rendered as text only when something reads it.
    /// Every node built through the DSL captures one, and the rendering — a reflection walk over
    /// every frame's method, its declaring type and its parameters — costs several times the rest
    /// of building the node, while nearly every graph has its call stacks stripped unread before
    /// it is exported. The text, when it is read, is exactly what
    /// <see cref="System.Diagnostics.StackTrace.ToString()"/> gives for the captured stack.
    /// </summary>
    internal sealed class CallStack
    {
        private readonly System.Diagnostics.StackTrace? _trace;
        private string? _text;

        private CallStack(System.Diagnostics.StackTrace? trace, string? text)
        {
            _trace = trace;
            _text = text;
        }

        /// <summary>The call stack of the caller, with file and line information, from the caller's
        /// own frame up.</summary>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        public static CallStack Capture() => new(new System.Diagnostics.StackTrace(skipFrames: 1, fNeedFileInfo: true), null);

        /// <summary>A call stack already rendered, or null for none.</summary>
        public static CallStack? FromText(string? text) => text is null ? null : new(null, text);

        /// <summary>The rendered call stack. Rendering is deterministic, so two threads reading it
        /// at once at worst render it twice and agree.</summary>
        public string Text
        {
            get
            {
                var text = _text;
                if (text is null)
                {
                    text = _trace!.ToString();
                    _text = text;
                }
                return text;
            }
        }
    }
}

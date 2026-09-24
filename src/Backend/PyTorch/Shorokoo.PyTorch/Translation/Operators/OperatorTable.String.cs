namespace Shorokoo.PyTorch.Translation.Operators;

// Strings and text; the semantics are in shorokoo_torch/ops_string.py. None of them carries a
// gradient: text has none, and TfIdfVectorizer counts.
internal static partial class OperatorTable
{
    static partial void RegisterString(Registry table)
    {
        const string M = "ops_string.";
        const TorchGradient none = TorchGradient.NotDifferentiable;
        table.Map("StringConcat", M + "string_concat", gradient: none);
        table.Map("StringSplit", M + "string_split", ["delimiter", "maxsplit"], tuple: true, gradient: none);
        table.Map("StringNormalizer", M + "string_normalizer",
            ["case_change_action", "is_case_sensitive", "locale", "stopwords"], gradient: none);
        table.Map("RegexFullMatch", M + "regex_full_match", ["pattern"], gradient: none);
        table.Map("TfIdfVectorizer", M + "tf_idf_vectorizer",
            ["max_gram_length", "max_skip_count", "min_gram_length", "mode", "ngram_counts", "ngram_indexes",
             "pool_int64s", "pool_strings", "weights"], gradient: none);
    }
}

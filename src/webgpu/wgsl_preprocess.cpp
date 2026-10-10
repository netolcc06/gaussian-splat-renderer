#include "webgpu/wgsl_preprocess.h"

#include "webgpu/wgsl_sources.h"

#include <set>

namespace gsr {
namespace {

constexpr int kMaxIncludeDepth = 32;

std::string_view trim(std::string_view s) {
    const size_t b = s.find_first_not_of(" \t\r");
    if (b == std::string_view::npos) return {};
    const size_t e = s.find_last_not_of(" \t\r");
    return s.substr(b, e - b + 1);
}

bool isIdentifier(std::string_view s) {
    if (s.empty()) return false;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        const bool alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
        const bool digit = c >= '0' && c <= '9';
        if (!alpha && !(digit && i > 0)) return false;
    }
    return true;
}

// One #ifdef/#ifndef block on the condition stack.
struct Conditional {
    bool parentActive;  // was the code around the block active?
    bool condition;     // value of the #ifdef / #ifndef test
    bool inElse;        // after #else
    uint32_t line;      // line of the opening directive (for "missing #endif")
    bool active() const { return parentActive && (inElse ? !condition : condition); }
};

class Preprocessor {
public:
    Preprocessor(std::span<const std::string> defines, const WgslFileLoader& loader)
        : defines_(defines.begin(), defines.end()), loader_(loader) {}

    bool run(std::string_view rootFile, WgslPreprocessResult& out) {
        out_ = &out;
        return processFile(rootFile, 0, "(root)", 0);
    }

private:
    bool fail(std::string_view file, uint32_t line, const std::string& message) {
        out_->error = std::string(file) + ":" + std::to_string(line) + ": " + message;
        return false;
    }

    bool processFile(std::string_view name, int depth, std::string_view fromFile, uint32_t fromLine) {
        if (depth > kMaxIncludeDepth) return fail(fromFile, fromLine, "#include nested too deeply");
        if (!included_.insert(std::string(name)).second) return true;  // implicit include guard
        const std::optional<std::string_view> text = loader_(name);
        if (!text) return fail(fromFile, fromLine, "cannot find included file \"" + std::string(name) + "\"");

        std::vector<Conditional> stack;
        auto active = [&]() { return stack.empty() || stack.back().active(); };
        uint32_t lineNo = 0;
        size_t pos = 0;
        while (pos <= text->size()) {
            size_t end = text->find('\n', pos);
            if (end == std::string_view::npos) end = text->size();
            const std::string_view line = text->substr(pos, end - pos);
            const bool lastLine = end == text->size();
            pos = end + 1;
            ++lineNo;
            if (lastLine && line.empty()) break;  // file ends with a newline

            const std::string_view t = trim(line);
            if (t.empty() || t[0] != '#') {
                if (active()) emit(line, name, lineNo);
                continue;
            }

            // Directive: "#word rest".
            const size_t wordEnd = std::min(t.find_first_of(" \t"), t.size());
            const std::string_view word = t.substr(1, wordEnd - 1);
            const std::string_view rest = trim(t.substr(wordEnd));
            if (word == "ifdef" || word == "ifndef") {
                if (!isIdentifier(rest)) return fail(name, lineNo, "#" + std::string(word) + " needs one identifier");
                const bool defined = defines_.count(std::string(rest)) != 0;
                stack.push_back({active(), word == "ifdef" ? defined : !defined, false, lineNo});
            } else if (word == "else") {
                if (stack.empty() || stack.back().inElse) return fail(name, lineNo, "#else without #ifdef/#ifndef");
                if (!rest.empty()) return fail(name, lineNo, "unexpected text after #else");
                stack.back().inElse = true;
            } else if (word == "endif") {
                if (stack.empty()) return fail(name, lineNo, "#endif without #ifdef/#ifndef");
                if (!rest.empty()) return fail(name, lineNo, "unexpected text after #endif");
                stack.pop_back();
            } else if (word == "define") {
                if (!isIdentifier(rest)) {
                    return fail(name, lineNo, "#define takes one identifier (flags only, no values)");
                }
                if (active()) defines_.insert(std::string(rest));
            } else if (word == "include") {
                if (rest.size() < 2 || rest.front() != '"' || rest.back() != '"') {
                    return fail(name, lineNo, "#include needs a quoted file name");
                }
                if (active() && !processFile(rest.substr(1, rest.size() - 2), depth + 1, name, lineNo)) return false;
            } else {
                return fail(name, lineNo, "unknown directive #" + std::string(word));
            }
        }
        if (!stack.empty()) return fail(name, stack.back().line, "missing #endif");
        return true;
    }

    void emit(std::string_view line, std::string_view file, uint32_t lineNo) {
        out_->code.append(line);
        out_->code.push_back('\n');
        out_->lineMap.push_back({std::string(file), lineNo});
    }

    std::set<std::string> defines_;
    std::set<std::string> included_;
    const WgslFileLoader& loader_;
    WgslPreprocessResult* out_ = nullptr;
};

}  // namespace

std::string WgslPreprocessResult::describeLine(uint64_t outputLine) const {
    if (outputLine == 0 || outputLine > lineMap.size()) return "?:" + std::to_string(outputLine);
    const WgslLineOrigin& o = lineMap[outputLine - 1];
    return o.file + ":" + std::to_string(o.line);
}

WgslPreprocessResult preprocessWgsl(std::string_view rootFile, std::span<const std::string> defines,
                                    const WgslFileLoader& loader) {
    WgslPreprocessResult result;
    Preprocessor pp(defines, loader);
    result.ok = pp.run(rootFile, result);
    if (!result.ok) {
        result.code.clear();
        result.lineMap.clear();
    }
    return result;
}

WgslFileLoader embeddedWgslLoader() {
    return [](std::string_view name) -> std::optional<std::string_view> {
        const WgslSourceFile* f = findEmbeddedWgsl(name);
        if (!f) return std::nullopt;
        return f->text;
    };
}

}  // namespace gsr

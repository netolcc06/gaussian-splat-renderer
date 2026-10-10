// The few platform-specific helpers of the UI's "Open scene..." feature, isolated here so
// that application.cpp stays portable: the native file-open dialog (Windows: the common
// item dialog, IFileOpenDialog; elsewhere a stub that reports "unsupported"), the
// executable's directory (where the recent-scenes list is kept) and UTF-8 <-> path
// conversions (GLFW and ImGui speak UTF-8; Windows paths are UTF-16).
#pragma once

#include <filesystem>
#include <string>
#include <string_view>

struct GLFWwindow;

namespace gsr {

struct FileDialogResult {
    enum class Status { Selected, Cancelled, Unsupported, Error };
    Status status = Status::Cancelled;
    std::filesystem::path path;  // Selected only
    std::string error;           // Unsupported / Error: a message for the UI
};

// Modal "open file" dialog filtered to *.ply (plus "All files"), parented to `window`
// and starting in `initialDirectory` if it exists. Blocks until the user closes it.
FileDialogResult openPlyFileDialog(GLFWwindow* window, const std::filesystem::path& initialDirectory = {});

// Directory of the running executable (falls back to the current directory).
std::filesystem::path executableDirectory();

// UTF-8 text <-> filesystem path, without the ANSI code page that std::string paths use
// on Windows (path::string() throws for names the code page cannot represent).
std::string pathToUtf8(const std::filesystem::path& path);
std::filesystem::path pathFromUtf8(std::string_view utf8);

}  // namespace gsr

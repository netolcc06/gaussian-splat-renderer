#include "app/file_dialog.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shobjidl.h>  // IFileOpenDialog (common item dialog, Windows Vista+)
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>  // glfwGetWin32Window: the dialog's owner window
#endif

#include <cstdio>
#include <system_error>

namespace gsr {

std::string pathToUtf8(const std::filesystem::path& path) {
    const std::u8string u8 = path.u8string();
    return std::string(u8.begin(), u8.end());
}

std::filesystem::path pathFromUtf8(std::string_view utf8) {
    return std::filesystem::path(std::u8string(utf8.begin(), utf8.end()));
}

#ifdef _WIN32

namespace {

// Releases a COM interface pointer at scope exit.
template <typename T>
struct ComPtr {
    T* p = nullptr;
    ~ComPtr() {
        if (p) p->Release();
    }
    T** operator&() { return &p; }
    T* operator->() const { return p; }
};

std::string hresultText(const char* what, HRESULT hr) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%s failed (HRESULT 0x%08lX)", what, static_cast<unsigned long>(hr));
    return buf;
}

}  // namespace

FileDialogResult openPlyFileDialog(GLFWwindow* window, const std::filesystem::path& initialDirectory) {
    FileDialogResult result;
    // The dialog is a COM object and wants a single-threaded apartment on this thread.
    // S_FALSE = COM was already initialized here (still needs the matching uninitialize).
    const HRESULT init = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    if (FAILED(init) && init != RPC_E_CHANGED_MODE) {
        result.status = FileDialogResult::Status::Error;
        result.error = hresultText("CoInitializeEx", init);
        return result;
    }
    {
        ComPtr<IFileOpenDialog> dialog;
        HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog));
        if (FAILED(hr)) {
            result.status = FileDialogResult::Status::Error;
            result.error = hresultText("Creating the file dialog", hr);
        } else {
            const COMDLG_FILTERSPEC filters[] = {{L"3D Gaussian Splatting scene (*.ply)", L"*.ply"},
                                                 {L"All files (*.*)", L"*.*"}};
            dialog->SetFileTypes(2, filters);
            dialog->SetFileTypeIndex(1);
            dialog->SetDefaultExtension(L"ply");
            dialog->SetTitle(L"Open scene");
            FILEOPENDIALOGOPTIONS options = 0;
            dialog->GetOptions(&options);
            dialog->SetOptions(options | FOS_FORCEFILESYSTEM | FOS_FILEMUSTEXIST | FOS_PATHMUSTEXIST);
            std::error_code ec;
            if (!initialDirectory.empty() && std::filesystem::is_directory(initialDirectory, ec)) {
                ComPtr<IShellItem> folder;
                const std::filesystem::path absolute = std::filesystem::absolute(initialDirectory, ec);
                if (!ec && SUCCEEDED(SHCreateItemFromParsingName(absolute.c_str(), nullptr, IID_PPV_ARGS(&folder))))
                    dialog->SetFolder(folder.p);
            }
            HWND owner = window ? glfwGetWin32Window(window) : nullptr;
            hr = dialog->Show(owner);  // modal: returns when the user picks a file or cancels
            if (hr == HRESULT_FROM_WIN32(ERROR_CANCELLED)) {
                result.status = FileDialogResult::Status::Cancelled;
            } else if (FAILED(hr)) {
                result.status = FileDialogResult::Status::Error;
                result.error = hresultText("The file dialog", hr);
            } else {
                ComPtr<IShellItem> item;
                PWSTR name = nullptr;
                hr = dialog->GetResult(&item);
                if (SUCCEEDED(hr)) hr = item->GetDisplayName(SIGDN_FILESYSPATH, &name);
                if (SUCCEEDED(hr) && name) {
                    result.status = FileDialogResult::Status::Selected;
                    result.path = std::filesystem::path(name);
                } else {
                    result.status = FileDialogResult::Status::Error;
                    result.error = hresultText("Reading the selected file", hr);
                }
                CoTaskMemFree(name);
            }
        }
    }
    if (SUCCEEDED(init)) CoUninitialize();
    return result;
}

std::filesystem::path executableDirectory() {
    std::wstring buffer(MAX_PATH, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(nullptr, buffer.data(), DWORD(buffer.size()));
        if (n == 0) break;
        if (n < buffer.size()) {
            buffer.resize(n);
            return std::filesystem::path(buffer).parent_path();
        }
        buffer.resize(buffer.size() * 2);  // truncated: long path
    }
    std::error_code ec;
    return std::filesystem::current_path(ec);
}

#else  // !_WIN32: no native dialog in this project (drag and drop and --load-after still work)

FileDialogResult openPlyFileDialog(GLFWwindow*, const std::filesystem::path&) {
    FileDialogResult result;
    result.status = FileDialogResult::Status::Unsupported;
    result.error = "No native file dialog on this platform: drop a .ply onto the window instead";
    return result;
}

std::filesystem::path executableDirectory() {
    std::error_code ec;
    const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec) return self.parent_path();
    return std::filesystem::current_path(ec);
}

#endif

}  // namespace gsr

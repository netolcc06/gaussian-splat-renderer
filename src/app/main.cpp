// Entry point: parse the command line, run the application, map failures to exit codes.
//   0 = success, 1 = runtime error, 2 = bad command line,
//   3 = automated run finished but the validation layer reported errors/warnings.
#include "app/application.h"
#include "app/cli.h"

#include <cstdio>
#include <exception>

int main(int argc, char** argv) {
    int exitCode = 0;
    const auto options = gsr::parseCommandLine(argc, argv, exitCode);
    if (!options) return exitCode;

    try {
        gsr::Application app(*options);
        return app.run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
}

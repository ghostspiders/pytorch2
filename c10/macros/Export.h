#ifndef C10_MACROS_EXPORT_H_
#define C10_MACROS_EXPORT_H_


#ifdef _WIN32
#if defined(C10_BUILD_SHARED_LIBS)
#define C10_EXPORT __declspec(dllexport)
#define C10_IMPORT __declspec(dllimport)
#else
#define C10_EXPORT
#define C10_IMPORT
#endif
#else // _WIN32
#if defined(__GNUC__)
#define C10_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define C10_EXPORT
#endif // defined(__GNUC__)
#define C10_IMPORT C10_EXPORT
#endif // _WIN32

// Definition of an adaptive XX_API macro, that depends on whether you are
// building the library itself or not, routes to XX_EXPORT and XX_IMPORT.
// Basically, you will need to do this for each shared library that you are
// building, and the instruction is as follows: assuming that you are building
// a library called libawesome.so. You should:
// (1) for your cmake target (usually done by "add_library(awesome, ...)"),
//     define a macro called AWESOME_BUILD_MAIN_LIB using
//     target_compile_options.
// (2) define the AWESOME_API macro similar to the one below.
// And in the source file of your awesome library, use AWESOME_API to
// annotate public symbols.

// Here, for the C10 library, we will define the macro C10_API for both import
// and export.

// This one is being used by libc10.so
#ifdef C10_BUILD_MAIN_LIB
#define C10_API C10_EXPORT
#else
#define C10_API C10_IMPORT
#endif

// This one is being used by libcaffe2.so
#ifdef CAFFE2_BUILD_MAIN_LIB
#define CAFFE2_API C10_EXPORT
#else
#define CAFFE2_API C10_IMPORT
#endif

#endif // C10_MACROS_MACROS_H_

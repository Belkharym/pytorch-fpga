#include "OpenCLException.h"

namespace c10 {
namespace opencl {
const char* clErrorString(cl_int error)
{
    static const char* strings[] =
    {
        // Error Codes
          "CL_SUCCESS"                                  //   0
        , "CL_DEVICE_NOT_FOUND"                         //  -1
        , "CL_DEVICE_NOT_AVAILABLE"                     //  -2
        , "CL_COMPILER_NOT_AVAILABLE"                   //  -3
        , "CL_MEM_OBJECT_ALLOCATION_FAILURE"            //  -4
        , "CL_OUT_OF_RESOURCES"                         //  -5
        , "CL_OUT_OF_HOST_MEMORY"                       //  -6
        , "CL_PROFILING_INFO_NOT_AVAILABLE"             //  -7
        , "CL_MEM_COPY_OVERLAP"                         //  -8
        , "CL_IMAGE_FORMAT_MISMATCH"                    //  -9
        , "CL_IMAGE_FORMAT_NOT_SUPPORTED"               //  -10
        , "CL_BUILD_PROGRAM_FAILURE"                    //  -11
        , "CL_MAP_FAILURE"                              //  -12

        , ""    //  -13
        , ""    //  -14
        , ""    //  -15
        , ""    //  -16
        , ""    //  -17
        , ""    //  -18
        , ""    //  -19

        , ""    //  -20
        , ""    //  -21
        , ""    //  -22
        , ""    //  -23
        , ""    //  -24
        , ""    //  -25
        , ""    //  -26
        , ""    //  -27
        , ""    //  -28
        , ""    //  -29

        , "CL_INVALID_VALUE"                            //  -30
        , "CL_INVALID_DEVICE_TYPE"                      //  -31
        , "CL_INVALID_PLATFORM"                         //  -32
        , "CL_INVALID_DEVICE"                           //  -33
        , "CL_INVALID_CONTEXT"                          //  -34
        , "CL_INVALID_QUEUE_PROPERTIES"                 //  -35
        , "CL_INVALID_COMMAND_QUEUE"                    //  -36
        , "CL_INVALID_HOST_PTR"                         //  -37
        , "CL_INVALID_MEM_OBJECT"                       //  -38
        , "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"          //  -39
        , "CL_INVALID_IMAGE_SIZE"                       //  -40
        , "CL_INVALID_SAMPLER"                          //  -41
        , "CL_INVALID_BINARY"                           //  -42
        , "CL_INVALID_BUILD_OPTIONS"                    //  -43
        , "CL_INVALID_PROGRAM"                          //  -44
        , "CL_INVALID_PROGRAM_EXECUTABLE"               //  -45
        , "CL_INVALID_KERNEL_NAME"                      //  -46
        , "CL_INVALID_KERNEL_DEFINITION"                //  -47
        , "CL_INVALID_KERNEL"                           //  -48
        , "CL_INVALID_ARG_INDEX"                        //  -49
        , "CL_INVALID_ARG_VALUE"                        //  -50
        , "CL_INVALID_ARG_SIZE"                         //  -51
        , "CL_INVALID_KERNEL_ARGS"                      //  -52
        , "CL_INVALID_WORK_DIMENSION"                   //  -53
        , "CL_INVALID_WORK_GROUP_SIZE"                  //  -54
        , "CL_INVALID_WORK_ITEM_SIZE"                   //  -55
        , "CL_INVALID_GLOBAL_OFFSET"                    //  -56
        , "CL_INVALID_EVENT_WAIT_LIST"                  //  -57
        , "CL_INVALID_EVENT"                            //  -58
        , "CL_INVALID_OPERATION"                        //  -59
        , "CL_INVALID_GL_OBJECT"                        //  -60
        , "CL_INVALID_BUFFER_SIZE"                      //  -61
        , "CL_INVALID_MIP_LEVEL"                        //  -62
        , "CL_INVALID_GLOBAL_WORK_SIZE"                 //  -63
        , "CL_UNKNOWN_ERROR_CODE"
    };

    if (error >= -63 && error <= 0)
         return strings[-error];
    else
         return strings[64];
}

const char* clDeviceTypeString(cl_device_type device_type) {
    static constexpr const char* strings[] =
    {
        // Device Types
          "UNKNOWN"                             //  0
        , "DEFAULT"                             //  1
        , "CPU"                                 //  2
        , "DEFAULT CPU"                         //  3
        , "GPU"                                 //  4
        , "DEFAULT GPU"                         //  5
        , "CPU GPU"                             //  6
        , "DEFAULT CPU GPU"                     //  7
        , "ACCELERATOR"                         //  8
        , "DEFAULT ACCELERATOR"                 //  9
        , "CPU ACCELERATOR"                     // 10
        , "DEFAULT CPU ACCELERATOR"             // 11
        , "GPU ACCELERATOR"                     // 12
        , "DEFAULT GPU ACCELERATOR"             // 13
        , "CPU GPU ACCELERATOR"                 // 14
        , "DEFAULT CPU GPU ACCELERATOR"         // 15
        , "CUSTOM"                              // 16
        , "DEFAULT CUSTOM"                      // 17
        , "CPU CUSTOM"                          // 18
        , "DEFAULT CPU CUSTOM"                  // 19
        , "GPU CUSTOM"                          // 20
        , "DEFAULT GPU CUSTOM"                  // 21
        , "CPU GPU CUSTOM"                      // 22
        , "DEFAULT CPU GPU CUSTOM"              // 23
        , "ACCELERATOR CUSTOM"                  // 24
        , "DEFAULT ACCELERATOR CUSTOM"          // 25
        , "CPU ACCELERATOR CUSTOM"              // 26
        , "DEFAULT CPU ACCELERATOR CUSTOM"      // 27
        , "GPU ACCELERATOR CUSTOM"              // 28
        , "DEFAULT GPU ACCELERATOR CUSTOM"      // 29
        , "CPU GPU ACCELERATOR CUSTOM"          // 30
        , "DEFAULT CPU GPU ACCELERATOR CUSTOM"  // 31
        , "ALL"
    };
    // CL_DEVICE_TYPE_DEFAULT                      (1 << 0)
    // CL_DEVICE_TYPE_CPU                          (1 << 1)
    // CL_DEVICE_TYPE_GPU                          (1 << 2)
    // CL_DEVICE_TYPE_ACCELERATOR                  (1 << 3)
    // CL_DEVICE_TYPE_CUSTOM                       (1 << 4)
    // CL_DEVICE_TYPE_ALL                          0xFFFFFFFF
    if (device_type < 32 /*(1 << 5)*/ && device_type >= 0) {
        return strings[device_type];
    }
    else {
        return strings[32];
    }
}
}} // namespace c10::opencl

# One executable per test; each returns 0 when every required check passes. They need RPP's library and an AMD GPU.
function(fastrpp_add_test NAME SOURCE)
    add_executable(${NAME} ${SOURCE})
    target_include_directories(${NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${NAME} PRIVATE FastRPP::FastRPP rpp::rpp)
    fastrpp_setup_target(${NAME})
    add_test(NAME ${NAME} COMMAND ${NAME})
endfunction()

fastrpp_add_test(fastrpp_compile_test basic/fastrpp_compile_test.hip)
fastrpp_add_test(fastrpp_errors_test errors/fastrpp_errors_test.hip)
fastrpp_add_test(fastrpp_resize_mirror_normalize_vs_rpp_test resize_mirror_normalize/fastrpp_resize_mirror_normalize_vs_rpp_test.hip)
fastrpp_add_test(fastrpp_channel_permute_vs_rpp_test channel_permute/fastrpp_channel_permute_vs_rpp_test.hip)
fastrpp_add_test(fastrpp_chain_vs_rpp_test chain/fastrpp_chain_vs_rpp_test.hip)

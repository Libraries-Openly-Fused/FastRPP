# Measurement tools: built with the library, not run by ctest (they time the GPU; run them only in a measurement slot).
add_executable(fastrpp_compare_implementations compare_implementations.hip)
target_link_libraries(fastrpp_compare_implementations PRIVATE FastRPP::FastRPP rpp::rpp)
fastrpp_setup_target(fastrpp_compare_implementations)

file(REMOVE_RECURSE
  "libhybridpricer.a"
  "libhybridpricer.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/hybridpricer.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()

#include "example_02.h"
#include <cstdio>
#include <vector>

int main() {
  float a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  float b[] = {2, 3, 4, 5, 6, 7, 8, 9, 10};
  float c[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  vec_add(a, b, c, sizeof(a));

  for (auto i : c) {
    printf("%.3f ", i);
  }
  printf("\n");
  return 0;
}
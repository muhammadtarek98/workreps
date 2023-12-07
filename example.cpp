#include <torch/torch.h>
#include <iostream>
using namespace torch;
using namespace std;

int main() {
	Tensor tensor = torch::rand({ 2, 3 });
	auto device = torch::cuda::is_available();
	tensor = tensor.cuda();

	cout << tensor << endl;
	return 0;
}

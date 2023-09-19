#include<torch/torch.h>
#include<iostream>
#include<cmath>
using namespace std;
void create_scalar_tensor() {
	torch::Tensor w = torch::tensor(1.0, torch::requires_grad());
	torch::Tensor b = torch::tensor(2.0, torch::requires_grad());
	cout << w << endl;
	cout << b;
}
void gradient_computation_linear_function() {
	torch::Tensor w = torch::tensor(1.0, torch::requires_grad());
	torch::Tensor b = torch::tensor(2.0, torch::requires_grad());
	torch::Tensor x = torch::tensor(3.0, torch::requires_grad());
	auto y = w * x + b;
	y.backward();
	cout << w.grad() << endl;
	cout << b.grad() << endl;
	cout << x.grad() << endl;
}
void autograd_with_optimizer_of_mlp() {
	//randn(shape of the tensor)
	torch::Tensor x = torch::randn({10,3});
	torch::Tensor y = torch::randn({10,2});
	//Linear(in_features,out_features)
	auto fc = torch::nn::Linear(3, 2);
	cout << fc->weight<< endl;
	cout<<fc->bias<<endl;
	//define the loss function
	torch::nn::MSELoss loss_function;
	//define the optimizer with initialized learning rate 0.01
	torch::optim::SGD optimizer(fc->parameters(), torch::optim::SGDOptions(0.01));
	//make prediction with forward propagtion
	auto pred = fc->forward(x);
	// calculate the loss
	auto loss = loss_function(pred, y);
	cout << "step 0" << endl;
	cout << loss.item<double>() << endl;
	//make backprop
	loss.backward();
	//print gradient before the optimizer update the parameters
	cout << fc->weight.grad() << endl;
	cout << fc->bias.grad() << endl;
	//make optimizer step
	optimizer.step();
	// print the gradient after 1-step 
	pred = fc->forward(x);
	loss = loss_function(pred, y);
	cout << "step 1" << endl;
	cout << loss.item<double>()<<endl;
	cout << fc->weight.grad() << endl;
	cout << fc->bias.grad() << endl;

}
void convert_from_arr_or_vec_to_tensor() {
	// convert array to tensor
	float arr[] = { 1,2,3,4 };
	//from_blob({shape of the tensor->(d1,d2,d3,d4,........dn)})
	torch::Tensor t1d = torch::from_blob(arr, { 1,4 });//vector 1D
	torch::Tensor t2d= torch::from_blob(arr, { 2,2 });//matrix 2D
	cout << t1d << endl;
	cout << t2d << endl;
	//convert vector to tensor
	vector<float>v = { 1,2,3,4 };
	torch::Tensor tv1d = torch::from_blob(v.data(), { 1,4 });
	torch::Tensor tv2d = torch::from_blob(v.data(), { 2,2 });
	cout << tv1d << endl;
	cout << tv2d << endl;
}
int main() {
	//create_scalar_tensor();
	//autograd();
	//autograd_with_optimizer_of_mlp();
	convert_from_arr_or_vec_to_tensor();

	return 0;
}
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
delta1=delta2=d2=d3=0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

J1=rtheta1=rtheta2=rtheta=0;
X = [ones(m, 1) X];
Theta11=transpose(Theta1);
Theta22=transpose(Theta2);

z1= X*Theta11;
a1=sigmoid(z1);
a2=[ones(m, 1) a1];
z2=a2*Theta22;
h=sigmoid(z2);

y1= eye(num_labels)(y,:);

for i=1:m
for k=1:num_labels
J1=J1+(-1)*((log(h(i,k))*y1(i,k))+(log(1-h(i,k))*(1-y1(i,k))));
endfor
endfor
J=(1/m)*J1;

s1=size(Theta1,1);
s2=size(Theta1,2);
s3=size(Theta2,1);
s4=size(Theta2,2);

for i=1:s1
for j=2:s2
rtheta1= rtheta1+power(Theta1(i,j),2);
endfor
endfor

for i=1:s3
for j=2:s4
rtheta2= rtheta2+power(Theta2(i,j),2);
endfor
endfor

rtheta=rtheta1+rtheta2;

J=J+((lambda/(2*m))*rtheta);

a1=X;
z2= a1*Theta11;
a2=sigmoid(z2);
a2=[ones(size(a2,1), 1) a2];
z3=a2*Theta22;
a3=sigmoid(z3);

d3=a3-y1;

g=sigmoidGradient(z2);
d2=(d3*Theta2(:,2:end)).*g;

delta1= transpose(d2)*a1;
delta2= transpose(d3)*a2;

Theta1_grad= delta1/m;
Theta2_grad= delta2/m;

Theta1(:,1)=0;
Theta2(:,1)=0;

Theta1_grad= (delta1/m)+((lambda/m)*Theta1);
Theta2_grad= (delta2/m)+((lambda/m)*Theta2);
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

function [cpu,gpu_notransfer,gpu_withtransfer] = sine_test(numrepeats,num_elems,opt)
%SINE_TEST benchmarks elementwise calculation of sine on CPU and GPU
% usage: [cpu,gpu_notransfer,gpu_withtransfer] = sine_test(numrepeats,num_elements,opt]
%
% If opt is empty or set to 'mean', sine_test will return the mean for each
% case.  Any other value for opt will result in all test values being
% returned.
%
% OUTPUTS:
% cpu - CPU timings
% gpu_notransfer - GPU timings without considering data transfer between host and GPU
% gpu_withtransfer - GPU timings including data transfer between host and GPU

if nargin<2
    error('usage: [cpu,gpu_notransfer,gpu_withtransfer] = sine_test(numrepeats,num_elements,opt]');
end

%If no option has been chosen, set default
if nargin<3
 opt = 'mean';
end

cpu = zeros(1,numrepeats);
gpu_notransfer = zeros(1,numrepeats);
gpu_withtransfer = zeros(1,numrepeats);

for i=1:numrepeats
  cpu_x = rand(1,num_elems)*10*pi;
  tic;cpu_y = sin(cpu_x);
  cpu(i) = toc;
end

for i=1:numrepeats
  cpu_x = rand(1,num_elems)*10*pi;
  %transfer to GPU
  gpu_x = gpuArray(cpu_x);
  %Do calculation
  tic;gpu_y = sin(gpu_x);
  gpu_notransfer(i) = toc;
  %Get result from GPU. Not included in timing here
  x = gather(gpu_y);
end

for i=1:numrepeats
  cpu_x = rand(1,num_elems)*10*pi;
  %transfer to GPU
  tic;
  gpu_x = gpuArray(cpu_x);
  %Do calculation
  gpu_y = sin(gpu_x);
  %Get result from GPU and put in main memory.  Now included in timing.
  x = gather(gpu_y);
  gpu_withtransfer(i) = toc;
end

if strcmp(opt,'mean')
  cpu = mean(cpu);
  gpu_notransfer = mean(gpu_notransfer);
  gpu_withtransfer = mean(gpu_withtransfer);
end

end



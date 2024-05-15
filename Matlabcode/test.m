z = 1:50;

nbytes = fprintf('processing 0 of %d', length(z));
for nz = z
    while nbytes > 0
         fprintf('\b')
         nbytes = nbytes - 1;
    end
  nbytes = fprintf('processing %d of %d', nz, length(z));
  % YOUR PROCESS HERE
  %
  %
  pause(1)
end
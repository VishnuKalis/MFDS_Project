% reading images
clc;
clear all;
images = zeros(64*64,10,15);
for k = 1:15
    chr = int2str(k);
    folder = strcat('Dataset_Question1\',chr);
    files = dir(folder);
    files = files(~[files.isdir]);
    for i = 1:10
        im = imread(fullfile(folder, files(i).name));
        images(:,i,k) = reshape(im, [], 1);
        images(:,i,k) = (images(:,i,k)-mean(images(:,i,k)))./std(images(:,i,k ));
    end
end

%SVD: Eigen Vectors and Eigen Values, Representative images
rep_images = zeros(4096,1,15);
for i = 1:15
    D = images(:,:,i);
    [U,u] = eig(D'*D);
    rep_images(:,1,i) = D*U(:,10) + D*U(:,9);
end
 
% Comparing and Checking
N  = zeros(1,15);
correct_identification = zeros(15,1);
for i = 1:15 % for fixing person
    count = 0;
    for j = 1:10 % for fixing condition
        for k = 1:15 % for comparing with all rep_images
            Img = images(:,j,i);
             Rep = rep_images(:,1,k);
             N(k) = norm(Img-Rep);
        end
         min = N(1);
         for z = 1:15 % for finding least norm
             if N(z)<min
                 min = N(z);
             end
         end
         if min == N(i) % checking recognition with corresponding image
             count = count+1;
         end
    end
    correct_identification(i,1)  = count;% number of images of a person identified correctly out of 10
end

Accuracy = sum(correct_identification)*100/150; % pecentage of images identified correctly out of 150

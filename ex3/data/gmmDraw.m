function gmmDraw (gmm, data)
% Usage gmmDraw( gmm, data )
%   gmm - array of gmm structs (see ex sheet)
%   data - matrix with data #dim x #samples
%
%   Plots the current state of the gmm model and
%   colors the data accordingly

K = length(gmm);
N = size( data, 2 );
ccc = 'bcrkmygbcrkmygbcrkmygbcrkmygbcrkmyg';

% find the strongest component for each data point
dists = zeros( K, N );
for i = 1:K
  d = data - repmat( gmm(i).mean, 1, N );
  dists(i,:) = sum( (inv( gmm(i).covm ) * d) .* d, 1);
end
[mindist, comp] = min( dists );

for j = 1:K
  plot(data(1,find(comp==j)), data(2,find(comp==j)), [ ccc(j) 'x' ])
  hold on
  
  plot(gmm(j).mean(1), gmm(j).mean(2), 'ko');
  
  [ U, L, V ] = svd(gmm(j).covm);
  
  
  phi = acos(V(1, 1));
  if (V(2, 1) < 0)
    phi = 2*pi - phi;
  end	
  
  h = ellipse(...
      2*sqrt(L(1, 1)), 2*sqrt(L(2,2)), phi, ...
      gmm(j).mean(1), gmm(j).mean(2), 'k' ...
		);
  
  set(h, 'LineWidth', 2); 
end

hold off

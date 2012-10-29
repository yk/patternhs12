function samples = gmmSample( gmm, N )
% Draws a few samples from a Gaussian Mixture Model
%
  samples = zeros( size(gmm(1).mean,1), N );
  ksum = cumsum( [gmm(:).p] );
  for i = 1:N
    % get the component
    kr = rand(1,1);
    k = min(find(kr<=ksum));
    
    % draw the sample
    samples(:,i) = mvnrnd( gmm(k).mean', gmm(k).covm )';

  end

end

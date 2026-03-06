% Changer le nombre de lignes et de colonnes d'une matrice
function [NewMat] = matrix(OldMat,line,colon)
if colon==-1
    if floor(size(OldMat,1)*size(OldMat,2)/line)==(size(OldMat,1)*...
            size(OldMat,2)/line);
        colon = size(OldMat,1)*size(OldMat,2)/line;
        NewMat=zeros(line,colon);
        NewMat(:)=OldMat(:);
    else
        colon = ceil(size(OldMat,1)*size(OldMat,2)/line);
        NewMat=zeros(line,colon);
        NewMat(1:(size(OldMat,1)*size(OldMat,2)))=OldMat(:);
    end
else
    if (line*colon)~=(size(OldMat,1)*size(OldMat,2));
        disp('Wrong new dimensions in function matrix');
    else
        NewMat=zeros(line,colon);
        NewMat(:)=OldMat(:);
    end
end
end

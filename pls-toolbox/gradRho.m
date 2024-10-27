function model = gradRho(model)

param       = model.initialParam;
beta        = 0.9;
beta2       = 0.5;


for i = 1:model.iter

    for j = 1:length(param)
        model.history(j).paraHist(i) = param(j);
    end

    if model.momentum ~= 2
        [loss, grad_mu] = dlfeval(@rhoAverage, dlarray(param), dlarray(model.X), dlarray(model.Y), dlarray(model.nsamp), model);
    else
        if i == 1
            [loss, grad_mu] = dlfeval(@rhoAverage, dlarray(param), dlarray(model.X), dlarray(model.Y), dlarray(model.nsamp), model);
        else
            a = length(model.history);
            b = [];
            for o = 1:a
                b = [b; model.history(o).paraHist(i-1)];
            end
            [loss, grad_mu] = dlfeval(@rhoAverage, dlarray(param - beta2 .* (param - b)), dlarray(model.X), dlarray(model.Y), dlarray(model.nsamp), model);
        end
    end

    for j = 1:length(param)

        if model.momentum == 1 % Polyak
            %                 momentum = beta*momentum + grad_mu(j);
            %                 param(j) = param(j) + model.learnRate.*momentum;
            % if model.loss == 1
            %     if i == 1
            %         param(j) = param(j) + model.learnRate.*(grad_mu(j));
            %     else
            %         param(j) = param(j) + model.learnRate.*(grad_mu(j)) + ...
            %             beta * (param(j) - model.history(j).paraHist(i-1));
            %     end
            % else
                if i == 1
                    param(j) = param(j) - model.learnRate.*(grad_mu(j));
                else
                    param(j) = param(j) - model.learnRate.*(grad_mu(j)) + ...
                        beta * (param(j) - model.history(j).paraHist(i-1));
                end
            %end

        elseif model.momentum == 2

            % if model.loss == 1
            %     if i == 1
            %         param(j) = param(j) + model.learnRate.*(grad_mu(j));
            %     else
            %         param(j)  = param(j) + beta2 * (param(j) - model.history(j).paraHist(i-1)) + model.learnRate.* grad_mu(j);
            %     end
            % else
                if i == 1
                    param(j) = param(j) + model.learnRate.*(grad_mu(j));
                else
                    param(j)  = param(j) + beta2 * (param(j) - model.history(j).paraHist(i-1)) - model.learnRate.* grad_mu(j);
                end

            %end


        else
            param(j) = param(j) + model.learnRate.*(grad_mu(j));
        end

        model.history(j).grad_muHist(i) = extractdata(grad_mu(j));
    end

    model.history(1).rhoHist(i) = extractdata(loss);
    %model.history(1).rhoHist(i) = loss;

end

end
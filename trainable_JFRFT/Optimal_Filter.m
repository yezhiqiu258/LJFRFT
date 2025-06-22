

function h_opt = Optimal_Filter(transform_mtx, joint_jfrt_mtx, joint_ijfrt_mtx, ...
                                    corr_xx, corr_xn, corr_nx, corr_nn)
    arguments
        transform_mtx(:, :) {mustBeNumeric, Must_Be_Square_Matrix}
        joint_jfrt_mtx(:, :) {mustBeNumeric, Must_Be_Square_Matrix, ...
                         Must_Be_Equal_Size(joint_jfrt_mtx, transform_mtx)}
        joint_ijfrt_mtx(:, :) {mustBeNumeric, Must_Be_Square_Matrix, ...
                          Must_Be_Equal_Size(joint_ijfrt_mtx, joint_jfrt_mtx)}
        corr_xx(:, :) {mustBeNumeric, Must_Be_Equal_Size(corr_xx, joint_jfrt_mtx)}
        corr_xn(:, :) {mustBeNumeric, Must_Be_Equal_Size(corr_xn, corr_xx)}
        corr_nx(:, :) {mustBeNumeric, Must_Be_Equal_Size(corr_nx, corr_xx)}
        corr_nn(:, :) {mustBeNumeric, Must_Be_Equal_Size(corr_nn, corr_xx)}
    end

    T = zeros(size(joint_jfrt_mtx));
    q = zeros(size(joint_jfrt_mtx, 1), 1);

    for m = 1:size(joint_jfrt_mtx, 1)
        wm            = joint_ijfrt_mtx(:, m);
        wm_tilde_T    = joint_jfrt_mtx(m, :);
        wm_tilde_conj = joint_jfrt_mtx(m, :)';

        q(m) = trace((transform_mtx' * wm) * (wm_tilde_T * corr_xx) + ...
                     (wm_tilde_conj * (wm' * corr_xn)));

        for n = 1:size(joint_jfrt_mtx, 2)
            wn          = joint_ijfrt_mtx(:, n);
            wn_tilde_T  = joint_jfrt_mtx(n, :);

            term1 = (transform_mtx' * wm_tilde_conj) * ...
              ((wn_tilde_T * transform_mtx) * (corr_xx + corr_nx));
            term2 = wm_tilde_conj * (((wn_tilde_T * transform_mtx) * corr_xn) + ...
                                     (wn_tilde_T * corr_nn));
            T(m, n) = (wm' * wn) * trace(term1 + term2);
        end
    end

    h_opt = T \ q;
end
def PRMS_loss(y_true, y_pred, scale=None):
    mseloss = torch.nn.MSELoss()
    if scale is not None:
        # rescale outputs before computing loss
        y_true = y_true.squeeze() / scale
        y_pred = y_pred.squeeze() / scale
    loss = mseloss(y_true.squeeze(), y_pred.squeeze())
    loss = torch.sqrt(loss)
    # get percent RMSE
    loss = loss * 100.0
    return loss


def relative_PRMS(y_true, y_pred, scale=None):
    # ignore scale param
    # take spatial average
    scale_rel = (y_pred**2).mean(dim=(-3, -2, -1), keepdim=True).sqrt()

    # upscale important of higher-norm instances
    return PRMS_loss(y_true / scale_rel, y_pred / scale_rel, scale=None)


def H1_loss(y_true, y_pred, scale=None, deriv_scale=10):
    # resid = y_true - y_pred

    diff_resid = central_diff_3d(y_true - y_pred)
    # stack along a new batch dimension (for now, will quickly get summed out)
    diff_resid = torch.stack(diff_resid, dim=1)

    L2_loss = PRMS_loss(y_true, y_pred, scale=scale)
    diff_loss = PRMS_loss(diff_resid, 0 * diff_resid, scale=scale) / deriv_scale

    # print(f"L2 is {L2_loss}, diff is {diff_loss}")

    return L2_loss + diff_loss


def compute_losses(model, quants_pred, quants_true, resid):

    strain_pred, stress_pred, energy_pred = quants_pred
    strain_true, stress_true, energy_true = quants_true

    strain_loss = H1_loss(
        strain_true,
        strain_pred,
        scale=model.strain_scaling,
        deriv_scale=model.config.H1_deriv_scaling,
    )
    stress_loss = H1_loss(
        stress_true,
        stress_pred,
        scale=model.stress_scaling,
        deriv_scale=model.config.H1_deriv_scaling,
    )
    energy_loss = H1_loss(
        energy_true,
        energy_pred,
        scale=model.energy_scaling,
        deriv_scale=model.config.H1_deriv_scaling,
    )

    resid_loss = 0
    stressdiv_loss = 0

    if model.config.return_resid:
        resid_loss = 100 * (resid**2).mean().sqrt() / model.strain_scaling

    if model.config.compute_stressdiv:
        stressdiv_loss = (
            100
            * (stressdiv(stress_pred, use_FFT_deriv=True) ** 2).mean().sqrt()
            / model.stress_scaling
        )

    losses = LossSet(
        model.config, strain_loss, stress_loss, energy_loss, resid_loss, stressdiv_loss
    )

    return losses.detach(), losses.compute_total()

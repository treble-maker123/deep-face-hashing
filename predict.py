import torch


def predict(model, loader_gallery, loader_test, logger, **kwargs):
    # moving model to CPU because GPU doesn't have enough memory
    device = kwargs.get("device", torch.device("cpu"))
    print_iter = kwargs.get("print_iter", 100)

    model.to(device=device)
    # set model to evaluation mode
    model.eval()

    # [gallery_codes, gallery_label, test_codes, test_label]
    data = [None] * 4

    with torch.no_grad():
        # process the gallery images
        logger.write("Hashing {} gallery images..."
                        .format(len(loader_gallery.dataset)))
        for idx, (X, y) in enumerate(loader_gallery):
            gcodes, _ = model(X.to(device=device))

            if data[0] is None:
                data[0] = gcodes
            else:
                data[0] = torch.cat((data[0], gcodes))

            if data[1] is None:
                data[1] = y
            else:
                data[1] = torch.cat((data[1], y))

            if idx % print_iter == 0:
                logger.write("{}/{} gallery batches completed..." \
                                .format(idx, len(loader_gallery)))

        assert len(loader_gallery.dataset) == len(data[0])
        assert len(loader_gallery.dataset) == len(data[1])

        logger.write("Hashing test images and labels...")
        # process the test images
        for idx, (X, y) in enumerate(loader_test):
            tcodes, _ = model(X.to(device=device))

            if data[2] is None:
                data[2] = tcodes
            else:
                data[2] = torch.cat((data[2], tcodes))

            if data[3] is None:
                data[3] = y
            else:
                data[3] = torch.cat((data[3], y))

            if idx % print_iter == 0:
                logger.write("{}/{} test batches completed..." \
                                .format(idx, len(loader_test)))

        gallery_codes, gallery_label, test_codes, test_label = data
        # activating with sign function
        bin_gallery_codes = gallery_codes > 0
        bin_test_codes = test_codes > 0

        # reshape labels so gallery and test match shape
        gallery_label = gallery_label.unsqueeze(1)
        test_label = test_label.unsqueeze(1)
        gallery_label = gallery_label.repeat(1, test_label.shape[0])
        test_label = test_label.repeat(1, gallery_label.shape[0])

        return bin_gallery_codes, gallery_label, bin_test_codes, test_label

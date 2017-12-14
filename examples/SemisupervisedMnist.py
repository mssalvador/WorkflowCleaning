import shared.create_dummy_data as ccd

def mnist():
    train_pdf, test_pdf = ccd.load_mnist()
    return train_pdf, test_pdf
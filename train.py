from Training.training import train_dt_model, train_rf_model, train_xgb_model


def main():
    train_dt_model()
    train_rf_model()
    train_xgb_model()


if __name__ == "__main__":
    main()

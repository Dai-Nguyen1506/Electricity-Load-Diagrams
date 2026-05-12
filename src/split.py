from sklearn.preprocessing import OrdinalEncoder


def split_train_test(
    df,
    target,
    split_time,
    categorical_cols=None
):

    train_df = df.loc[:split_time].copy()
    test_df = df.loc[split_time:].iloc[1:].copy()

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    encoder = None

    if categorical_cols:

        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        X_train[categorical_cols] = encoder.fit_transform(
            X_train[categorical_cols]
        )

        X_test[categorical_cols] = encoder.transform(
            X_test[categorical_cols]
        )

    return X_train, X_test, y_train, y_test, encoder
import time

from Util import k_fold_cross_validation, create_dataset, get_model, plot_heatmap_print_acc

for part in ["A", "B"]:
    start_time = time.time()
    print("================Part " + part + "================")
    # data = create_dataset(1000, part=part)
    # for h1 in [1, 4, 8]:
    #     for h2 in [0, 3]:
    #         model_start = time.time()
    #         print("======================================")
    #
    #         ###############################################
    #         # part a
    #         result = k_fold_cross_validation(h1, h2, data)
    #         print("For h1 =", h1, "and h2 =", h2)
    #         print("Accuracy =", str(round(result["acc"], 1)), "+-", str(round(result["acc_se"], 1)))
    #         print("ROC AUC =", str(round(result["roc_auc"], 1)), "+-", str(round(result["roc_auc_se"], 1)))
    #         ###############################################
    #         print("======================================")
    #         ###############################################
    #         # part b
    #         model = get_model(h1, h2)
    #         model.fit(*data, epochs=100, verbose=0)
    #         plot_heatmap_print_acc(model, part, filename="heatmaps/" + part + str(h1) + "_" + str(h2) + ".jpg")
    #         del model
    #         ###############################################
    #
    #         print("This model done in", time.time() - model_start)
    #         print("======================================")

    #######################################################
    # part c
    print("======================================")
    data = create_dataset(10000, part=part)
    result = k_fold_cross_validation(12, 3, data)
    print("Accuracy =", str(round(result["acc"], 1)), "+-", str(round(result["acc_se"], 1)))
    print("ROC AUC =", str(round(result["roc_auc"], 1)), "+-", str(round(result["roc_auc_se"], 1)))
    model = get_model(12, 3)
    model.fit(*data, epochs=100, verbose=0)
    plot_heatmap_print_acc(model, part=part, filename="heatmaps/" + part + "1c_12_3.jpg")

    print("======================================")

    data = create_dataset(10000, part=part)
    result = k_fold_cross_validation(24, 9, data)
    print("Accuracy =", str(round(result["acc"], 1)), "+-", str(round(result["acc_se"], 1)))
    print("ROC AUC =", str(round(result["roc_auc"], 1)), "+-", str(round(result["roc_auc_se"], 1)))
    model = get_model(24, 9)
    model.fit(*data, epochs=100, verbose=0)
    plot_heatmap_print_acc(model, part=part, filename="heatmaps/" + part + "1c_24_9.jpg")
    #######################################################

    print("Total time", time.time() - start_time)

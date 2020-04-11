import time

from Util import k_fold_cross_validation_ensemble, create_dataset

for part in ["A", "B"]:
    # for decision in ["majority_vote", "average"]:
        print("======================================")
        print("======================================")
        decision = "average"
        print("Part", part, "Decision", decision)
        start_time = time.time()
        data = create_dataset(1000, part=part)
        for h1 in [1, 4, 8]:
            for h2 in [0, 3]:
                model_start = time.time()
                print("======================================")
                print("For h1 =", h1, "and h2 =", h2)

                ###############################################
                # part a
                result = k_fold_cross_validation_ensemble(h1, h2, data, decision=decision, sample_train=False)
                print("Accuracy =", str(round(result["acc"], 1)), "+-", str(round(result["acc_se"], 1)))
                if decision != "majority_vote":
                    print("ROC AUC =", str(round(result["roc_auc"], 1)), "+-", str(round(result["roc_auc_se"], 1)))
                ###############################################

                ###############################################
                # part b
                result = k_fold_cross_validation_ensemble(h1, h2, data, decision=decision, sample_train=True)
                print("=============Train Sampled============")
                print("Accuracy =", str(round(result["acc"], 1)), "+-", str(round(result["acc_se"], 1)))
                if decision != "majority_vote":
                    print("ROC AUC =", str(round(result["roc_auc"], 1)), "+-", str(round(result["roc_auc_se"], 1)))
                ###############################################

                print("This model done in", time.time() - model_start)
                print("======================================")

        print("Total time", time.time() - start_time)
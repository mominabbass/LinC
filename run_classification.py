import argparse
import torch
from data_utils import load_dataset_custom
from utils import *
import torch.nn as nn
from torch.autograd import Variable
from sklearn.utils import shuffle
from scipy.stats import entropy
from matplotlib import pyplot as plt

def main(models, datasets, all_shots, num_seeds, subsample_test_set, api_num_log_prob, approx, use_saved_results, bs,
         lr, val_seed, val_size, epochs):
    """
    Run experiment or load past results, print accuracy
    """
    default_params = {
        'conditioned_on_correct_classes': True,
        'subsample_test_set': subsample_test_set,
        'val_size': val_size,
        'lr': lr,
        'epochs': epochs,
        'val_seed': val_seed,
        'api_num_log_prob': api_num_log_prob,
        'approx': approx,
        'bs': bs
    }

    # list of all experiment parameters to run
    all_params = []
    for model in models:
        for dataset in datasets:
            for num_shots in all_shots:
                for seed in range(num_seeds):
                    p = deepcopy(default_params)
                    p['model'] = model
                    p['dataset'] = dataset
                    p['seed'] = seed
                    p['num_shots'] = num_shots
                    p[
                        'expr_name'] = f"{p['dataset']}_{p['model']}_{p['num_shots']}shot_{repr(p['subsample_test_set'])}_subsample_seed{p['seed']}"
                    all_params.append(p)

    # query the model and save the responses
    if use_saved_results:
        load_results(all_params)
    else:
        save_results(all_params)

def save_results(params_list, freeze_test_set=True):
    """
    Run the model and save its responses and the rest of configs into a pickle file
    """

    result_tree = dict()
    for param_index, params in enumerate(params_list):
        print("\nExperiment name:", params['expr_name'])

        val_size = params['val_size']
        lr = params['lr']
        epochs = params['epochs']
        curr_seed = params['val_seed']

        ### load data
        all_train_sentences, all_train_labels, all_test_sentences, all_test_labels, all_val_sentences, all_val_labels = load_dataset_custom(
            params)

        ### sample test set
        if params['subsample_test_set'] is None:
            test_sentences, test_labels = all_test_sentences, all_test_labels
            print(f"selecting full test set ({len(all_test_labels)} examples)")
        else:
            if freeze_test_set:
                np.random.seed(0)  # always use seed 0 result if freeze
            else:
                np.random.seed(params['seed'])
            test_sentences, test_labels = random_sampling(all_test_sentences, all_test_labels,
                                                          params['subsample_test_set'])
            print(f"selecting {len(test_labels)} subsample of test set")

        ### sample validaion set
        if params['subsample_test_set'] is None:
            val_sentences, val_labels = all_val_sentences, all_val_labels
            print(f"selecting full validation set ({len(all_test_labels)} examples)")
        else:
            if freeze_test_set:
                np.random.seed(0)  # always use seed 0 result if freeze
            else:
                np.random.seed(params['seed'])
            val_sentences, val_labels = random_sampling(all_val_sentences, all_val_labels, val_size)
            print(f"selecting {len(val_labels)} subsample of validation set")

        ### sample few-shot training examples
        np.random.seed(params['seed'])
        train_sentences, train_labels = random_sampling(all_train_sentences, all_train_labels,
                                                        params['num_shots'])

        ### Evaluate the performance and save all results
        # obtaining model's response on test examples
        print(f"getting raw resp for {len(test_sentences)} test sentences")

        params_check(params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences,
                     val_labels, test_sentences, test_labels)
        # get prob for each label
        _, all_label_probs = get_model_response(params, all_train_sentences, all_train_labels, train_sentences,
                                                train_labels, val_sentences, val_labels, test_sentences, test_labels)

        # calculate P_cf
        content_free_inputs = ["N/A", "", "[MASK]"]
        p_cf = get_p_content_free(params, all_train_sentences, all_train_labels, train_sentences, train_labels,
                                  val_sentences, val_labels, test_labels,
                                  content_free_inputs=content_free_inputs)  ##type: numpy array e.g. [0.13829783 0.86170214] for SST2

        # acc_original, entropy_original, prob_original, ECE_original = eval_accuracy(all_label_probs, test_labels, all_train_sentences,
        #                                                all_train_labels,
        #                                                val_sentences, val_labels, params, lr, epochs, curr_seed)
        acc_calibrated, entropy_calibrated, prob_calibrated, ECE_calibrated = eval_accuracy(all_label_probs,
                                                                                            test_labels,
                                                                                            all_train_sentences,
                                                                                            all_train_labels,
                                                                                            val_sentences, val_labels,
                                                                                            params, lr, epochs,
                                                                                            curr_seed,
                                                                                            mode="diagonal_W",
                                                                                            p_cf=p_cf)
        accuracies = [acc_calibrated]
        ECE = [ECE_calibrated]
        print(f"Accuracies: {accuracies}")
        print(f"Calibration Errors (ECE): {ECE}")
        print(f"p_cf      : {p_cf}")

        # add to result_tree
        keys = [params['dataset'], params['model'], params['num_shots']]
        node = result_tree  # root
        for k in keys:
            if not (k in node.keys()):
                node[k] = dict()
            node = node[k]
        node[params['seed']] = accuracies

        # save to file
        result_to_save = dict()
        params_to_save = deepcopy(params)
        result_to_save['params'] = params_to_save
        result_to_save['train_sentences'] = train_sentences
        result_to_save['train_labels'] = train_labels
        result_to_save['test_sentences'] = test_sentences
        result_to_save['test_labels'] = test_labels
        result_to_save['all_label_probs'] = all_label_probs
        result_to_save['p_cf'] = p_cf
        result_to_save['accuracies'] = accuracies
        result_to_save['ECE'] = ECE
        if 'prompt_func' in result_to_save['params'].keys():
            params_to_save['prompt_func'] = None
        save_pickle(params, result_to_save)

    print_results(result_tree)


def eval_accuracy(all_label_probs, test_labels, all_train_sentences, all_train_labels, val_sentences, val_labels,
                  params, lr, epochs, curr_seed, mode=None, p_cf=None):
    # evaluate the accuracy with and without contextual calibration
    criterion = nn.CrossEntropyLoss()
    num_classes = all_label_probs.shape[1]
    uncalib = 0

    if p_cf is None:
        # do not calibrate
        W = Variable(torch.eye(num_classes), requires_grad=True)
        b = Variable(torch.zeros([num_classes, 1]), requires_grad=True)
        uncalib += 1
    else:
        np.random.seed(curr_seed)
        train_sentences, train_labels = random_sampling(all_train_sentences, all_train_labels, params['num_shots'])
        _, all_val_label_probs = get_model_response(params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, val_sentences, val_labels)

        # calibrate
        if mode == "diagonal_W":
            W = Variable(torch.inverse(torch.eye(num_classes) * torch.tensor(p_cf)), requires_grad=True)
            b = Variable(torch.zeros([num_classes, 1]), requires_grad=True)
        elif mode == "identity_W":
            W = Variable(torch.identity(num_classes), requires_grad=True)
            b = Variable(-1 * torch.unsqueeze(p_cf, axis=-1, requires_grad=True))
        else:
            assert False

        optimizer = torch.optim.SGD([W, b], lr=lr)

        val_labels = np.array(val_labels)

        for epoch in range(epochs):
            all_val_label_probs, val_labels = shuffle(all_val_label_probs, val_labels, random_state=0)

            for label_probs, true_label in zip(all_val_label_probs, val_labels):
                optimizer.zero_grad()
                label_probs = torch.tensor(label_probs) / torch.sum(torch.tensor(label_probs))  # normalize to 1

                calibrate_label_probs = torch.matmul(W.float(),
                                                     torch.unsqueeze(label_probs, dim=-1).float()) + b.float()

                loss = criterion(calibrate_label_probs.reshape(1, len(calibrate_label_probs)),
                                 torch.tensor(true_label).reshape(1))

                loss.backward()
                if not (torch.isnan(W.grad).any() or torch.isnan(b.grad).any()):
                    optimizer.step()

    entropy_list = []
    correctness_list = []
    prob_list = []
    sft_mx = nn.Softmax(dim=0)
    calib_prob = []

    assert len(all_label_probs) == len(test_labels)

    for label_probs, true_label in zip(all_label_probs, test_labels):
        label_probs = torch.tensor(label_probs) / torch.sum(torch.tensor(label_probs))  # normalize to 1

        calibrate_label_probs = torch.matmul(W.float(), torch.unsqueeze(label_probs, dim=-1).float()) + b.float()

        if(uncalib == 1):
            calib_prob.append(calibrate_label_probs[:,0].tolist())
        else:
            calib_prob.append(sft_mx(calibrate_label_probs)[:, 0].tolist())

        # calculate entropy
        if not (torch.isnan(calibrate_label_probs).any()):
            H = np.around(entropy(sft_mx(calibrate_label_probs).detach().numpy(), base=2), decimals=1)
            entropy_list.append(H)

        ans_label = torch.argmax(calibrate_label_probs)
        prob_list.append(round(sft_mx(calibrate_label_probs).detach().numpy()[true_label][0], 1))

        if ans_label == true_label:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
    entropy_list = np.array(entropy_list)

    ECE = expected_calibration_error(np.array(calib_prob), np.array(test_labels), M=10)[0]

    return np.mean(correctness_list), entropy_list, prob_list, ECE

def get_label_probs(params, raw_resp, train_sentences, train_labels, test_sentences):
    """Obtain model's label probability for each of the test examples. The returned prob is NOT normalized"""
    num_classes = len(params['label_dict'])
    approx = params['approx']
    assert len(raw_resp) == len(test_sentences)

    # Fill in the labels that is in the top k prob
    all_label_probs = []
    all_missing_positions = []
    for i, ans in enumerate(raw_resp):
        top_logprobs = ans['logprobs']['top_logprobs'][0]  # [0] since we only ask for complete one more token
        label_probs = [0] * len(params['label_dict'].keys())
        for j, label_list in params['label_dict'].items():
            all_found = True
            for label in label_list:  # each possible label correspond to the same class
                label = " " + label  # notice prompt does not have space after 'A:'
                if label in top_logprobs:
                    label_probs[j] += np.exp(top_logprobs[label])
                else:
                    all_found = False
            if not all_found:
                position = (i, j)  # (which test example, which label)
                all_missing_positions.append(position)
        all_label_probs.append(label_probs)
    all_label_probs = np.array(all_label_probs)  # prob not normalized

    # Fill in the label probs that are NOT in top k probs, by asking the model to rate perplexity
    # This helps a lot in zero shot as most labels wil not be in Top 100 tokens returned by LM
    if (not approx) and (len(all_missing_positions) > 0):
        print(f"Missing probs: {len(all_missing_positions)}/{len(raw_resp) * num_classes}")
        all_additional_prompts = []
        num_prompts_each = []
        for position in all_missing_positions:
            which_sentence, which_label = position
            test_sentence = test_sentences[which_sentence]
            label_list = params['label_dict'][which_label]
            for label in label_list:
                prompt = construct_prompt(params, train_sentences, train_labels, test_sentence)
                prompt += " " + label
                all_additional_prompts.append(prompt)
            num_prompts_each.append(len(label_list))

        # chunk the prompts and feed into model
        chunked_prompts = list(chunks(all_additional_prompts, chunk_size_helper(params)))
        all_probs = []
        for chunk_id, chunk in enumerate(chunked_prompts):
            resp = complete(chunk, 0, params['model'], echo=True, num_log_probs=1)
            for ans in resp['choices']:
                prob = np.exp(ans['logprobs']['token_logprobs'][-1])
                all_probs.append(prob)

        assert sum(num_prompts_each) == len(all_probs)
        assert len(num_prompts_each) == len(all_missing_positions)

        # fill in corresponding entries in all_label_probs
        for index, num in enumerate(num_prompts_each):
            probs = []
            while num > 0:
                probs.append(all_probs.pop(0))
                num -= 1
            prob = np.sum(probs)
            i, j = all_missing_positions[index]
            all_label_probs[i][j] = prob

        assert len(all_probs) == 0, "all should be popped"
        assert (all_label_probs > 0).all(), "all should be populated with non-zero value"

    return all_label_probs  # NOT NORMALIZED

def get_p_content_free(params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_labels, content_free_inputs=('N/A')):
    """Query model with content free input, return its prediction probability for each label"""

    _, all_p_y = get_model_response(params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, content_free_inputs, test_labels, normalize=False)

    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y)  # normalize
    return p_y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--models', dest='models', action='store', required=True,
                        help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True,
                        help='name of dataset(s), e.g., agnews')
    parser.add_argument('--num_seeds', dest='num_seeds', action='store', required=True,
                        help='num seeds for the training set', type=int)
    parser.add_argument('--all_shots', dest='all_shots', action='store', required=True,
                        help='num training examples to use')
    # LinC arguments
    parser.add_argument('--lr', dest='lr', action='store', required=True, help='learning rate alpha', type=float)
    parser.add_argument('--val_seed', dest='val_seed', action='store', required=True,
                        help='seed to select the random set of validation demonstrations', type=int)
    parser.add_argument('--epochs', dest='epochs', action='store', required=True, help='total numbr of epochs T',
                        type=int)
    parser.add_argument('--val_size', dest='val_size', action='store', required=True,
                        help='size of validation set i.e. number of validation prompts', type=int)
    # other arguments
    parser.add_argument('--subsample_test_set', dest='subsample_test_set', action='store', required=False, type=int,
                        default=None, help='size of test set to use to speed up eval. None means using all test set')
    parser.add_argument('--api_num_log_prob', dest='api_num_log_prob', action='store', required=False, type=int,
                        default=100,
                        help='number of top tokens to ask for when querying the model. Capped at 100 for OpenAI GPT-3 API')
    parser.add_argument('--bs', dest='bs', action='store', required=False, type=int, default=None,
                        help='batch size for model queries. For OpenAI API, capped at 20. For local running, set this to max out your GPU memory.')
    # flags
    parser.add_argument('--use_saved_results', dest='use_saved_results', action='store_const', const=True,
                        default=False,
                        help='whether to load the results from pickle files and not run the model')
    parser.add_argument('--approx', dest='approx', action='store_const', const=True, default=False,
                        help='whether to set token prob to zero if not in top 100')

    args = parser.parse_args()
    args = vars(args)


    # simple processing
    def convert_to_list(items, is_int=False):
        if is_int:
            return [int(s.strip()) for s in items.split(",")]
        else:
            return [s.strip() for s in items.split(",")]


    args['models'] = convert_to_list(args['models'])
    args['datasets'] = convert_to_list(args['datasets'])
    args['all_shots'] = convert_to_list(args['all_shots'], is_int=True)

    main(**args)
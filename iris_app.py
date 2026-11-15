"""Streamlit v1: the root project's Model owns all neural-network computation."""
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris, load_wine, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from model import Model
from backprop_inspector import backward_snapshot
from visualization_helpers import display_values, architecture_figure, layer_figure


def prepare_data(name):
    if name == 'Moons':
        X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
        features, classes = ['x1', 'x2'], ['Class 0', 'Class 1']
    else:
        data = {'Iris': load_iris, 'Wine': load_wine}[name]()
        X, y = data.data, data.target
        features, classes = list(data.feature_names), list(data.target_names)
    raw_train, raw_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler().fit(raw_train)
    train = scaler.transform(raw_train).tolist()
    test = scaler.transform(raw_test).tolist()
    labels = [[float(j == int(label)) for j in range(len(classes))] for label in y_train]
    return dict(train=train, test=test, labels=labels,
                y_train=y_train.tolist(), y_test=y_test.tolist(),
                raw_test=raw_test.tolist(), features=features, classes=classes)


def predictions(model, rows):
    probabilities = [display_values(model.forward(row)).tolist() for row in rows]
    if not all(math.isfinite(p) for row in probabilities for p in row):
        raise FloatingPointError('Non-finite predictions. Try a smaller learning rate or hidden size.')
    return probabilities, [max(range(len(row)), key=row.__getitem__) for row in probabilities]


def forward_snapshot(model, sample):
    """Capture the actual cache immediately after one Model.forward call."""
    probabilities = display_values(model.forward(sample)).tolist()
    steps = [dict(x=display_values(w.x).tolist(), xb=display_values(w.xb).tolist(), W=display_values(w.weight_matrix).tolist(),
                  z=display_values(w.z).tolist(), y=display_values(w.y).tolist(), activation=w.activation_function.__name__)
             for w in model.weights]
    return probabilities, steps


def main():
    st.set_page_config(page_title='PhysicsLLM | Neural Lab', layout='wide')
    st.title('Neural Lab')
    st.caption('Train and inspect your Vector / Matrix / Layer / Weight / Model implementation.')
    with st.sidebar:
        st.header('Training setup')
        dataset = st.selectbox('Dataset', ['Iris', 'Wine', 'Moons'])
        hidden = st.slider('Hidden size (both layers)', 2, 16, 8)
        epochs = st.number_input('Epochs', min_value=1, max_value=200, value=30, step=1)
        lr = st.select_slider('Learning rate', options=[0.0001, 0.001, 0.005, 0.01, 0.02, 0.05], value=0.01)
        train = st.button('Train model', type='primary')
        st.caption('A fresh Model is trained on each click. Inspection does not retrain it.')
    config = (dataset, hidden, int(epochs), lr)
    data = prepare_data(dataset)
    st.write(f"{len(data['features'])} inputs → {hidden} ReLU → {hidden} ReLU → {len(data['classes'])} logits → softmax")
    st.caption('80% training / 20% held out; stratified split. Scaling is fitted on training features only.')
    if train:
        st.session_state.pop('result', None)
        progress = st.progress(0.0)
        train_accuracy, validation_accuracy = [], []
        def report(epoch, loss):
            if not math.isfinite(loss):
                raise FloatingPointError('Training diverged. Try a smaller learning rate or hidden size.')
            _, epoch_train = predictions(model, data['train'])
            _, epoch_val = predictions(model, data['test'])
            train_accuracy.append(float(accuracy_score(data['y_train'], epoch_train)))
            validation_accuracy.append(float(accuracy_score(data['y_test'], epoch_val)))
            progress.progress(epoch / int(epochs), text=f'Epoch {epoch}/{epochs} · loss {loss:.5f}')
        try:
            model = Model(len(data['features']), hidden, len(data['classes']))
            with st.spinner('Training your Model with its existing SGD loop…'):
                history = model.train(data['train'], data['labels'], num_epochs=int(epochs),
                                      lr=lr, on_epoch=report)
                _, train_pred = predictions(model, data['train'])
                test_prob, test_pred = predictions(model, data['test'])
            st.session_state.result = dict(model=model, history=history, config=config,
                data=data, train_accuracy=train_accuracy, validation_accuracy=validation_accuracy, train_pred=train_pred, test_prob=test_prob, test_pred=test_pred)
        except (ValueError, FloatingPointError, OverflowError) as error:
            st.error(str(error))
            return
    result = st.session_state.get('result')
    if result is None:
        st.info('Choose a dataset and press Train model to begin.')
        return
    if result['config'] != config:
        st.info('Settings changed. Press Train model to train with these settings.')
        return
    if 'train_accuracy' not in result:
        st.info('Please retrain to collect per-epoch accuracy for this version.')
        return
    model, data = result['model'], result['data']
    training, architecture, weights, inspector, backprop = st.tabs(['Results', 'Architecture', 'Weight matrices', 'Math Inspector', 'Backprop Inspector'])
    with architecture:
        sizes = [model.fc1.weight_matrix.shape[0] - int(model.fc1.bias)] + [w.weight_matrix.shape[1] for w in model.weights]
        fig = architecture_figure(sizes)
        st.pyplot(fig)
        plt.close(fig)
        st.caption('Every circle is a unit. Arrows show dense connections. Bias rows are omitted here and shown in the weight matrices. Softmax follows the output logits.')
    with training:
        left, right = st.columns(2)
        left.metric('Training accuracy', f"{accuracy_score(data['y_train'], result['train_pred']):.1%}")
        right.metric('Held-out accuracy', f"{accuracy_score(data['y_test'], result['test_pred']):.1%}")
        st.line_chart(pd.DataFrame({'Epoch': range(1, len(result['history'])+1),
                                   'Cross-entropy': result['history']}).set_index('Epoch'))
        st.caption('Mean online training loss: weights change after each sample. This is not a held-out loss curve.')
        st.subheader('Accuracy vs epoch')
        st.line_chart(pd.DataFrame({'Epoch': range(1, len(result['history'])+1),
                                   'Training': result['train_accuracy'],
                                   'Validation': result['validation_accuracy']}).set_index('Epoch'))
        st.caption('Accuracy is evaluated after each epoch with fixed weights; validation samples never update weights.')
        rows = pd.DataFrame({'Actual': [data['classes'][i] for i in data['y_test']],
                             'Predicted': [data['classes'][i] for i in result['test_pred']]})
        for i, name in enumerate(data['classes']):
            rows[f'P({name})'] = [p[i] for p in result['test_prob']]
        st.dataframe(rows)
    with weights:
        st.caption('Rows are inputs, with the final row holding bias weights. Columns are output units.')
        for i, weight in enumerate(model.weights):
            values = display_values(weight.weight_matrix).tolist()
            limit = max(abs(v) for row in values for v in row) or 1.0
            fig, ax = plt.subplots(figsize=(6, 3))
            im = ax.imshow(values, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
            ax.set(title=f'fc{i+1}: {weight.weight_matrix.shape}', xlabel='Output unit', ylabel='Input / bias')
            ax.set_yticks(range(len(values)), [str(j) for j in range(len(values)-1)] + ['bias'])
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            with st.expander(f'Weight {i+1} numerical values'):
                st.dataframe(pd.DataFrame(values, index=[str(j) for j in range(len(values)-1)] + ['bias']))
    with inspector:
        st.caption('Input → weighted sum → ReLU → weighted sum → ReLU → output logits → softmax')
        index = st.slider('Held-out sample', 0, len(data['test'])-1, 0, key=f'sample_{dataset}')
        probabilities, steps = forward_snapshot(model, data['test'][index])
        st.write('Actual class: ' + data['classes'][data['y_test'][index]])
        st.dataframe(pd.DataFrame({'Feature': data['features'], 'Raw input': data['raw_test'][index],
                                  'Standardized input': data['test'][index]}))
        for i, step in enumerate(steps):
            st.subheader(f"fc{i+1} · {step['activation']}")
            fig = layer_figure(step)
            st.pyplot(fig)
            plt.close(fig)
            st.caption('Blue bars are input values; gold is the appended constant 1. Gray/teal bars compare the actual weighted sums and activations.')
            st.write('Input x (cached Vector)')
            st.dataframe(pd.DataFrame([step['x']]))
            st.write('Bias step: xb = [x, 1] (cached Vector)')
            st.dataframe(pd.DataFrame([step['xb']]))
            st.latex(r'z = x_b W')
            st.write(f"W shape: {len(step['W'])} × {len(step['W'][0])}; final row is bias")
            st.dataframe(pd.DataFrame(step['W'], index=[str(j) for j in range(len(step['W'])-1)] + ['bias']))
            st.write('Linear transformation z (cached Layer)')
            st.dataframe(pd.DataFrame([step['z']]))
            if i < 2:
                st.latex(r'y = \mathrm{ReLU}(z) = \max(0,z)')
                st.caption('Negative weighted sums become zero; positive values pass through.')
            else:
                st.latex(r'y = z')
                st.caption('The final activation is linear. These values are logits, not probabilities.')
            st.write('Activation y (cached Layer)')
            st.dataframe(pd.DataFrame([step['y']]))
        st.subheader('Softmax probabilities')
        st.latex(r'p_i = e^{z_i-\max(z)} / \sum_j e^{z_j-\max(z)}')
        st.bar_chart(pd.DataFrame({'Class': data['classes'], 'Probability': probabilities}).set_index('Class'))
        st.dataframe(pd.DataFrame({'Class': data['classes'], 'Logit': steps[-1]['y'], 'Probability': probabilities}))
        st.caption('Probabilities come directly from Model.forward(). Only caches are refreshed; model weights are unchanged.')

    with backprop:
        st.subheader('Backprop Inspector')
        st.caption('One sample, copied model, no weight updates. Gradients flow fc3 → fc2 → fc1.')
        selected = st.slider('Backprop sample', 0, len(data['test'])-1, 0, key=f'backprop_{dataset}')
        target = [float(i == data['y_test'][selected]) for i in range(len(data['classes']))]
        loss, gradients = backward_snapshot(model, data['test'][selected], target)
        st.metric('Sample cross-entropy', f'{loss:.6f}')
        st.write('Actual class: ' + data['classes'][data['y_test'][selected]])
        st.caption('Starting dy = p − target: the combined softmax + cross-entropy gradient with respect to logits. The output layer is linear.')
        magnitudes = pd.DataFrame([{'Layer': g['layer'],
            'dy L2': math.sqrt(sum(v*v for v in g['dy'])),
            'dz L2': math.sqrt(sum(v*v for v in g['dz'])),
            'dx L2': math.sqrt(sum(v*v for v in g['dx'])),
            'dW Frobenius': math.sqrt(sum(v*v for row in g['dW'] for v in row))}
            for g in gradients]).set_index('Layer')
        st.bar_chart(magnitudes)
        st.dataframe(magnitudes)
        st.caption('Magnitudes are norms for this sample, not epoch averages. dW includes the bias row.')
        for g in gradients:
            st.subheader(g['layer'])
            st.latex(r'dy \;\downarrow\; dz = f\prime(z) \odot dy \;\downarrow\; dx = \mathrm{unbias}(dz W^T)')
            fig, axes = plt.subplots(3, 1, figsize=(8, 5))
            for ax, name in zip(axes, ['dy', 'dz', 'dx']):
                ax.bar(range(len(g[name])), g[name], color=['#ef4444' if v < 0 else '#2563eb' for v in g[name]])
                ax.axhline(0, color='gray', linewidth=0.5)
                ax.set(ylabel=name, xlabel='Unit index')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(pd.DataFrame({'dy': g['dy'], "activation derivative": g['derivative'], 'dz': g['dz']}))
            st.write('dx: gradient passed to the previous layer (bias component removed)')
            st.dataframe(pd.DataFrame([g['dx']]))
            st.latex(r'dW = x_b^T dz')
            st.write('Cached xb')
            st.dataframe(pd.DataFrame([g['xb']]))
            limit = max(abs(v) for row in g['dW'] for v in row) or 1.0
            fig, ax = plt.subplots(figsize=(7, 3))
            im = ax.imshow(g['dW'], cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
            ax.set(title=g['layer'] + ' · dW', xlabel='Output unit', ylabel='Input / bias')
            row_labels = [str(i) for i in range(len(g['dW'])-1)] + ['bias']
            ax.set_yticks(range(len(row_labels)), row_labels)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(pd.DataFrame(g['dW'], index=row_labels))


if __name__ == '__main__':
    main()


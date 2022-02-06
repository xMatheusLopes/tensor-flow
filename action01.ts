import * as tf from '@tensorflow/tfjs';

console.log('Processando...');

const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

const x = tf.tensor([1, 2, 3, 4], [4, 1]);
const y = tf.tensor([10, 20, 30, 40], [4, 1]);
const input = tf.tensor([5, 6, 7], [3, 1]);

model.fit(x, y, {epochs: 550}).then(() => {
    let output = (model.predict(input) as tf.Tensor).dataSync()
    output = convertArray(output);
    let z = tf.tensor(output);

    console.log(z.toString());
});

function convertArray(array: any): Int32Array {
	let result = [];
	for(let i=0; i<array.length; i++) {
		result.push(Math.ceil(array[i]));
	}
	return Int32Array.from(result);
}
from classifier import shallow_neural_net
from planar_utils import load_planar_dataset

def train():

    X, Y = load_planar_dataset()
    model = shallow_neural_net(4, 1, X, Y)
    model(10000)

if __name__ == '__main__':
    train()
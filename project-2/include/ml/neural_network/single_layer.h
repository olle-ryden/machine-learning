//! @note Se kommentarer för motsvarande fil i assignment 5.
/**
 * @brief Single Layer implementation.
 */
#pragma once

#include "ml/neural_network/interface.h"
#include "ml/dense_layer/interface.h"

namespace ml::neural_network
{
/**
 * @brief Implement SingleLayer class by inheriting Interface.
 *
 * @note SingleLayer class should not be inheritable.
 */
class SingleLayer final: public Interface
{
public:
    /**
     * @brief Create new SingleLayer with a hidden layer.
     *
     * @param[in] hiddenLayer Reference to the neural network hidden layer, dense layer interface.
     * @param[in] outputLayer Reference to the neural network output layer, dense layer interface.
     * @param[in] trainInput Reference to readable-only vector containing floats.
     * @param[in] trainOutput Reference to readable-only vector containing floats.
     *
     * @note Jag är inte säker på hur jag ska beskriva 'hiddenLayer' och 'outputLayer'.
     */
    SingleLayer(dense_layer::Interface& hiddenLayer, dense_layer::Interface& outputLayer,
                const std::vector<std::vector<double>>& trainInput,
                const std::vector<std::vector<double>>& trainOutput) noexcept;

    /**
     * @brief Delete the default constructor.
     */
    ~SingleLayer() noexcept override = default;

    /**
     * @brief Method for implementing a prediction with the model.
     *
     * @param[in] input The input data to base the prediction on, as a reference to a vector
     *                  containing float numbers.
     *
     * @return The predicted value as a vector containing float numbers.
     */
    const std::vector<double>& predict(const std::vector<double>& input) override;

    /**
     * @brief Method for training the model.
     *
     * @param[in] epochCount an unsigned integer indicating the amount of training epochs.
     * @param[in] learningRate Learning speed as a float, default 0.01 (1% learning speed).
     *
     * @return Normalized precision (0.0 - 1.0) post training, or -1.0 on error.
     */
    double train(std::size_t epochCount, double learningRate = 0.01);

    /**
     * @brief Get the accuracy of the latest training.
     *
     * @return Floating number normalized to 0.0-1.0.
     */
    double accuracy() override;


    /**
     * @brief Get the average error of predicted values.
     *
     * @param[in] input Vector containing the input training data.
     * @param[in] reference Vector containing the expected values of the training data.
     */
    double averageError(const std::vector<double>& input,
                        const std::vector<double>& reference) override;

    /**
     * @brief Delete the default constructor, delete copy and move constructors, delete operators.
     */
    SingleLayer()                                 = delete;
    SingleLayer(const SingleLayer&)               = delete;
    SingleLayer(SingleLayer&&)                    = delete;
    SingleLayer& operator=(const SingleLayer&)    = delete;
    SingleLayer& operator=(SingleLayer&&)         = delete;

private:

    /** The hidden layer of the network. */
    dense_layer::Interface& myHiddenLayer;

    /** The output layer of the network. */
    dense_layer::Interface& myOutputLayer;

    /** The training input data. */
    const std::vector<std::vector<double>>& myTrainInput;

    /** The training output data. */
    const std::vector<std::vector<double>>& myTrainOutput;

    /** The amount of complete training sets, where there's the same amount of input/output data. */
    const std::size_t myTrainSetCount;

};
} // namespace ml::neural_network
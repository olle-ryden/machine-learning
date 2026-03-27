//! @note Se kommentarer för motsvarande fil i assignment 5.
/**
 * @brief Single Layer implementation details.
 */
#include <vector>

#include "ml/neural_network/single_layer.h"

namespace ml::neural_network
{
namespace
{
/**
 * @brief Method for making sure the training set count is not larger than the smallest value of
 *        either the input or the output.
 *
 * @param[in] input 2D vector containing float numbers.
 * @param[in] output 2D vector containing float numbers.
 *
 * @return The size of the smallest vector as an unsigned integer.
 */
std::size_t getTrainSetCount(const std::vector<std::vector<double>>& input,
                             const std::vector<std::vector<double>>& output) noexcept
{
    return (input.size() <= output.size() ? input.size() : output.size());
}

/**
 * @brief Get the absolute value of the given number.
 *
 * @param[in] num The number in question.
 *
 * @return The absolute value of the given number.
 */
constexpr double absVal(const double num) noexcept { return 0.0 <= num ? num : -num; }

} // namespace

//--------------------------------------------------------------------------------
SingleLayer::SingleLayer(dense_layer::Interface& hiddenLayer, dense_layer::Interface& outputLayer,
                         const std::vector<std::vector<double>>& trainInput,
                         const std::vector<std::vector<double>>& trainOutput) noexcept
    : myHiddenLayer(hiddenLayer)
    , myOutputLayer(outputLayer)
    , myTrainInput(trainInput)
    , myTrainOutput(trainOutput)
    , myTrainSetCount(getTrainSetCount(trainInput, trainOutput))
{}

//--------------------------------------------------------------------------------
const std::vector<double>& SingleLayer::predict(const std::vector<double>& input)
{
    myHiddenLayer.feedforward(input);
    myOutputLayer.feedforward(myHiddenLayer.output());
    return myOutputLayer.output();
}

//--------------------------------------------------------------------------------
double SingleLayer::train(std::size_t epochCount, double learningRate)
{
    // Return -1.0 if any faulty epoch count, train set or learning rate.
    if (0U == myTrainSetCount || 0U == epochCount || 0.0 >= learningRate) { return -1.0; }

    myHiddenLayer.initParams();
    myOutputLayer.initParams();

    // Train for 'epochCount' number of times.
    for (std::size_t epoch{}; epoch < epochCount; ++epoch)
    {
        for (std::size_t i = 0U; i < myTrainSetCount; ++i)
        {
            // Get the values of the current training set.
            const auto& input = myTrainInput[i];
            const auto& output = myTrainOutput[i];

            // Feedforward.
            myHiddenLayer.feedforward(input);
            myOutputLayer.feedforward(myHiddenLayer.output());

            // Backpropagate.
            myOutputLayer.backpropagate(output);
            myHiddenLayer.backpropagate(myOutputLayer);

            // Optimize.
            myOutputLayer.optimize(myHiddenLayer.output(), learningRate);
            myHiddenLayer.optimize(input, learningRate);
        }
    }
    // Return the accuracy as, normalize to 0.0 - 1.0.
    return accuracy();

}

// --------------------------------------------------------------------------------
double SingleLayer::accuracy()
{
    double sum{};

    for (std::size_t i{}; i < myTrainSetCount; ++i)
    {
        sum += averageError(myTrainInput[i], myTrainOutput[i]);
    }
    const auto avgError{sum / myTrainSetCount};
    const auto precision{1.0 - avgError};
    return precision;
}

// -----------------------------------------------------------------------------
double SingleLayer::averageError(const std::vector<double>& input,
                                 const std::vector<double>& reference)
{
    double error{};

    // Get predicted values.
    const auto prediction{predict(input)};

    // Compare predicted values with expected values (reference), accumulate all deviations.
    for (std::size_t i{}; i < prediction.size(); ++i)
    {
        error += absVal(reference[i] - prediction[i]);
    }
    return error / reference.size();
}
} // namespace ml::neural_network
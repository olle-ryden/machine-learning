//! \note Utmärkt interface med virtuella metoder för precision med mera.

/**
 * @brief Neural network interface implementation.
 */
#pragma once

#include <vector>

namespace ml::neural_network
{
/**
 * @brief Neural network interface implementation.
 */
class Interface
{
public:
    /**
     * @brief Delete the default destructor.
     */
    virtual ~Interface() noexcept = default;

    /**
     * @brief Virtual method for implementing a prediction with the model.
     *
     * @param[in] input The input data for basing the prediction upon.
     *
     * @return The predicted value as a vector containing float numbers.
     */
    virtual const std::vector<double>& predict(const std::vector<double>& input) = 0;

    /**
     * @brief Virtual method for getting the accuracy of the latest training.
     *
     * @return Floating number normalized to 0.0-1.0.
     */
    virtual double accuracy() = 0;

    /**
     * @brief Get the average error of predicted values.
     *
     * @param[in] input Vector containing the input training data.
     * @param[in] reference Vector containing the expected values of the training data.
     */
    virtual double averageError(const std::vector<double>& input,
                                const std::vector<double>& reference) = 0;
};
} // namespace ml::neural_network

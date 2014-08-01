local HingeEmbeddingCriterion, parent = torch.class('jzt.HingeEmbeddingCriterion','nn.Criterion')

function HingeEmbeddingCriterion:__init(margin)
   parent.__init(self)
   self.margin = margin
   self.output = torch.CudaTensor()
end

function HingeEmbeddingCriterion:updateOutput(input, target)
   self.output:resizeAs(input)
   jzt.HingeEmbeddingCriterion_updateOutput(input, target, self.output, self.margin)
   return self.output
end

function HingeEmbeddingCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   jzt.HingeEmbeddingCriterion_updateGradInput(input, target, self.gradInput, self.margin)
   return self.gradInput:div(input:size(1))
end

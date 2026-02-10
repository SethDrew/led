#ifndef TREE_EFFECT_H
#define TREE_EFFECT_H

#include "../TreeTopology.h"

// Blend modes for compositing layers
enum BlendMode {
  REPLACE,  // Background replaces previous content
  ADD,      // Additive blending (sum with cap at 255)
  ALPHA,    // Alpha blending (not yet implemented)
  MULTIPLY  // Multiply blending (not yet implemented)
};

// Base class for all tree effects
// Renders directly to Tree strips (no buffer overhead)
class TreeEffect {
protected:
  Tree* tree;

public:
  TreeEffect(Tree* t) : tree(t) {}
  virtual ~TreeEffect() {}

  // Update effect state (called each frame)
  virtual void update() = 0;

  // Render effect directly to tree
  virtual void render() = 0;
};

// Base class for background effects
class TreeBackgroundEffect : public TreeEffect {
public:
  TreeBackgroundEffect(Tree* t) : TreeEffect(t) {}
  virtual void render() = 0;
};

// Base class for foreground effects
class TreeForegroundEffect : public TreeEffect {
public:
  TreeForegroundEffect(Tree* t) : TreeEffect(t) {}
  virtual void render() = 0;
};

#endif

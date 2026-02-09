import { betterAuth } from "better-auth";

export const auth = betterAuth({
  // TODO: Configure database adapter for better-auth
  // For now, using memory adapter for development
  emailAndPassword: {
    enabled: true,
  },
  session: {
    expiresIn: 60 * 60 * 24 * 7, // 7 days
    updateAge: 60 * 60 * 24, // 1 day
  },
});

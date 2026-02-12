"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Plus, MessageSquare, Trash2, Calendar, Loader2 } from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar";
import {
  Dialog,
  DialogContent,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { getTrips, deleteTrip, type TripSummary } from "@/lib/api";
import type { AuthUser } from "@/lib/api";

interface ChatHistorySidebarProps {
  user?: AuthUser | null;
  loading?: boolean;
  onSignOut?: () => void;
}

function formatDate(dateString?: string): string {
  if (!dateString) return "";
  const date = new Date(dateString);
  const now = new Date();
  const diffInDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));
  
  if (diffInDays === 0) return "Today";
  if (diffInDays === 1) return "Yesterday";
  if (diffInDays < 7) return `${diffInDays} days ago`;
  const weeks = Math.floor(diffInDays / 7);
  if (diffInDays < 30) return `${weeks} ${weeks === 1 ? 'week' : 'weeks'} ago`;
  
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export function ChatHistorySidebar({
  user,
  loading,
}: ChatHistorySidebarProps) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { state } = useSidebar();
  const isCollapsed = state === "collapsed";
  const [trips, setTrips] = useState<TripSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [tripToDelete, setTripToDelete] = useState<TripSummary | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const activeTripId =
    searchParams.get("tripId") || searchParams.get("id");

  const fetchTrips = async () => {
    try {
      const data = await getTrips();
      // Sort by most recent first
      const sorted = data.sort((a, b) => {
        const dateA = new Date(a.last_message_at || a.updated_at).getTime();
        const dateB = new Date(b.last_message_at || b.updated_at).getTime();
        return dateB - dateA;
      });
      setTrips(sorted);
    } catch {
      setError("Unable to load trips.");
    }
  };

  useEffect(() => {
    if (loading || !user) return;
    fetchTrips();
  }, [loading, user]);

  const handleDeleteClick = (e: React.MouseEvent, trip: TripSummary) => {
    e.preventDefault();
    e.stopPropagation();
    setTripToDelete(trip);
  };

  const handleConfirmDelete = async () => {
    if (!tripToDelete) return;
    
    setIsDeleting(true);
    try {
      await deleteTrip(tripToDelete.id);
      setTrips((prev) => prev.filter((t) => t.id !== tripToDelete.id));
      if (activeTripId === tripToDelete.id) {
        router.push("/");
      }
      setTripToDelete(null);
    } catch {
      alert("Failed to delete trip.");
    } finally {
      setIsDeleting(false);
    }
  };

  if (loading || !user) return null;

  return (
    <>
      <Sidebar 
        collapsible="icon" 
        className="border-r border-border/50 bg-[#FAFAF8]"
      >
        {/* Header with Toggle - Right aligned when expanded, centered when collapsed */}
        <SidebarHeader className={`flex flex-row items-center p-4 border-b border-border/30 ${isCollapsed ? 'justify-center' : 'justify-end'}`}>
          <SidebarTrigger className="hover:bg-accent/10 transition-colors cursor-pointer" />
        </SidebarHeader>

        <SidebarContent className="px-2 py-4">
          <SidebarGroup>
            {/* New Plan Button */}
            <div className={`pb-4 flex ${isCollapsed ? 'justify-center px-0' : 'px-2'}`}>
              <button
                onClick={() => {
                  // Force a full page navigation to reset all state
                  window.location.href = "/trip";
                }}
                className={`
                  flex items-center bg-accent text-white hover:bg-accent/90 transition-all 
                  shadow-sm hover:shadow-md rounded-xl cursor-pointer
                  ${isCollapsed 
                    ? 'w-10 h-10 justify-center p-0 shrink-0' 
                    : 'w-full gap-3 px-3 py-3'
                  }
                `}
                title={isCollapsed ? "New Plan" : undefined}
              >
                <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-white/20 backdrop-blur-sm">
                  <Plus className="size-4" />
                </div>
                {!isCollapsed && (
                  <div className="flex flex-col gap-0.5 leading-none text-left">
                    <span className="font-semibold text-sm text-white">New Plan</span>
                    <span className="text-[10px] text-white/70">Start a journey</span>
                  </div>
                )}
              </button>
            </div>

          {/* Recent Trips Section */}
          {!isCollapsed && (
            <div className="px-2 pb-2">
              <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                <Calendar className="w-3 h-3" />
                <span className="font-body">Recent Trips</span>
              </div>
            </div>
          )}
            {isCollapsed && (
              <SidebarGroupLabel className="justify-center px-0">
                <Calendar className="w-4 h-4" />
              </SidebarGroupLabel>
            )}
            
            <SidebarGroupContent className="mt-2">
              <SidebarMenu className="gap-1">
                {error ? (
                  <SidebarMenuItem className={`text-sm text-muted-foreground bg-muted/50 rounded-lg ${isCollapsed ? 'p-2 text-center' : 'px-3 py-3'}`}>
                    {!isCollapsed && (
                      <div className="flex items-center gap-2">
                        <span className="text-red-500">●</span>
                        {error}
                      </div>
                    )}
                    {isCollapsed && <span className="text-red-500 text-xs">!</span>}
                  </SidebarMenuItem>
                ) : trips.length === 0 ? (
                  <SidebarMenuItem className={`text-sm text-muted-foreground bg-muted/30 rounded-lg ${isCollapsed ? 'p-2 text-center' : 'px-3 py-6 text-center'}`}>
                    <div className={`flex flex-col items-center gap-2 ${isCollapsed ? '' : ''}`}>
                      <MessageSquare className={`opacity-40 ${isCollapsed ? 'w-4 h-4' : 'w-5 h-5'}`} />
                      {!isCollapsed && (
                        <>
                          <span>No trips yet</span>
                          <span className="text-[10px] opacity-60">Create your first trip plan</span>
                        </>
                      )}
                    </div>
                  </SidebarMenuItem>
                ) : (
                  trips.map((trip) => {
                    const isActive = activeTripId === trip.id;
                    return (
                      <SidebarMenuItem key={trip.id} className="relative group/menu-item">
                        <button
                          onClick={() => router.push(`/trip?tripId=${trip.id}`)}
                          title={isCollapsed ? `${trip.origin} → ${trip.destination}` : undefined}
                          className={`
                            flex items-center transition-all duration-200 cursor-pointer
                            ${isActive 
                              ? "bg-white shadow-sm border border-border/50" 
                              : "hover:bg-white/50 hover:shadow-sm"
                            }
                            rounded-xl
                            ${isCollapsed 
                              ? 'w-10 h-10 justify-center p-0 mx-auto' 
                              : 'w-full gap-3 px-3 py-2.5'
                            }
                          `}
                        >
                          <div className={`
                            flex aspect-square size-8 items-center justify-center rounded-lg
                            ${isActive ? "bg-accent/10 text-accent" : "bg-muted text-muted-foreground"}
                          `}>
                            <MessageSquare className="size-4" />
                          </div>
                          {!isCollapsed && (
                            <div className="flex flex-col overflow-hidden min-w-0 text-left pr-8">
                              <span className={`truncate font-medium text-sm ${isActive ? "text-foreground" : "text-foreground/80"}`}>
                                {trip.origin} → {trip.destination}
                              </span>
                              <div className="flex items-center gap-2">
                                {trip.last_message && (
                                  <span className="truncate text-[11px] text-muted-foreground/70 max-w-[100px]">
                                    {trip.last_message}
                                  </span>
                                )}
                                <span className="text-[10px] text-accent/70 shrink-0">
                                  {formatDate(trip.last_message_at || trip.updated_at)}
                                </span>
                              </div>
                            </div>
                          )}
                        </button>
                        {!isCollapsed && (
                          <button
                            onClick={(e) => handleDeleteClick(e, trip)}
                            className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-md opacity-0 group-hover/menu-item:opacity-100 transition-opacity hover:bg-muted cursor-pointer z-10"
                          >
                            <Trash2 className="size-3.5 text-muted-foreground hover:text-destructive" />
                          </button>
                        )}
                      </SidebarMenuItem>
                    );
                  })
                )}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>
      </Sidebar>

      {/* Delete Confirmation Modal */}
      <Dialog open={!!tripToDelete} onOpenChange={() => !isDeleting && setTripToDelete(null)}>
        <DialogContent className="sm:max-w-lg w-sm p-0 overflow-hidden">
          {/* Trip Info Header */}
          <div className="bg-slate-50 px-8 py-6 border-b border-slate-100">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-xl bg-white border border-slate-200 flex items-center justify-center shrink-0 shadow-sm">
                <MessageSquare className="w-6 h-6 text-slate-400" />
              </div>
              <div className="overflow-hidden flex-1">
                <p className="text-base font-semibold text-foreground truncate">
                  {tripToDelete?.origin} → {tripToDelete?.destination}
                </p>
                <p className="text-sm text-muted-foreground">
                  {tripToDelete && formatDate(tripToDelete.created_at)}
                </p>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="p-6 flex gap-3">
            <Button
              variant="ghost"
              onClick={() => setTripToDelete(null)}
              disabled={isDeleting}
              className="flex-1 h-11 rounded-lg hover:bg-slate-50 cursor-pointer bg-slate-100"
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleConfirmDelete}
              disabled={isDeleting}
              className="flex-1 h-11 rounded-lg bg-red-500 hover:bg-red-600 cursor-pointer gap-2"
            >
              {isDeleting ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Trash2 className="w-4 h-4" />
              )}
              {isDeleting ? "Deleting..." : "Delete Trip"}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
